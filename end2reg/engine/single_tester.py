from typing import Dict
import json
import torch
import numpy as np
import os
import ipdb
import traceback
from tqdm import tqdm

from end2reg.engine.base_tester import BaseTester
from end2reg.utils.summary_board import SummaryBoard
from end2reg.utils.timer import Timer
from end2reg.utils.common import get_log_string
from end2reg.utils.torch import release_cuda, to_cuda


class SingleTester(BaseTester):
    def __init__(self, cfg, parser=None, cudnn_deterministic=True):
        super().__init__(cfg, parser=parser, cudnn_deterministic=cudnn_deterministic)

        self.transforms = []
        self.metadata = []
        self.segm_scores = []
        # Avoid WDDM spilling into shared GPU memory on Windows:
        # keep some headroom and trim the caching allocator only when needed.
        self._vram_trim_ratio = float(os.environ.get("GEOTR_VRAM_TRIM_RATIO", "0.90"))
        self._vram_limit_ratio = float(os.environ.get("GEOTR_VRAM_LIMIT_RATIO", "0.92"))

    def before_test_epoch(self):
        pass

    def before_test_step(self, iteration, data_dict):
        pass

    def test_step(self, iteration, data_dict) -> Dict:
        pass

    def eval_step(self, iteration, data_dict, output_dict) -> Dict:
        pass

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        est = output_dict['estimated_transform'].cpu().numpy()
        metadata = output_dict.get('metadata', None)
        segm_out = output_dict['scores_seg'].cpu()
        # scores = torch.softmax(segm_out, dim=1)[:, 1]
        scores = torch.softmax(segm_out, dim=1)
        self.transforms.append(est)
        # IMPORTANT: ensure we don't keep CUDA tensors alive via metadata references.
        self.metadata.append(release_cuda(metadata) if metadata is not None else None)
        self.segm_scores.append(scores.numpy())

    def after_test_epoch(self):
        os.makedirs('predictions', exist_ok=True)
        transforms_array = np.stack(self.transforms)  # shape (N, 4, 4)
        metadata_array = np.array(self.metadata, dtype=object)
        scores_array = np.array(self.segm_scores, dtype=object)
        np.savez('predictions/results.npz',
                 transforms=transforms_array,
                 metadata=metadata_array,
                 segm_scores=scores_array)

        print("Saved all transformations and metadata in results.npz")


    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        return get_log_string(result_dict)

    def run(self):
        assert self.test_loader is not None
        self.load_snapshot(self.args.snapshot)
        self.model.eval()
        if self.args.compile and hasattr(self.model, 'compile_submodules'):
            self.model.compile_submodules()
        torch.set_grad_enabled(False)
        # Keep VRAM headroom to prevent Windows WDDM shared-memory fallback.
        try:
            torch.cuda.set_per_process_memory_fraction(self._vram_limit_ratio)
        except Exception:
            pass
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        self.before_test_epoch()
        summary_board = SummaryBoard(adaptive=True)
        timer = Timer()
        total_iterations = len(self.test_loader)
        pbar = tqdm(enumerate(self.test_loader), total=total_iterations)
        for iteration, data_dict in pbar:
            # on start
            self.iteration = iteration + 1
            data_dict = to_cuda(data_dict)
            try:
                self.before_test_step(self.iteration, data_dict)
                # test step
                torch.cuda.synchronize()
                timer.add_prepare_time()
                with torch.inference_mode():
                    output_dict = self.test_step(self.iteration, data_dict)
                if output_dict is None:
                    continue
                torch.cuda.synchronize()
                timer.add_process_time()
                # eval step
                with torch.inference_mode():
                    result_dict = self.eval_step(self.iteration, data_dict, output_dict)
                # after step
                self.after_test_step(self.iteration, data_dict, output_dict, result_dict)
                # logging
                result_dict = release_cuda(result_dict)
                summary_board.update_from_result_dict(result_dict)
                message = self.summary_string(self.iteration, data_dict, output_dict, result_dict)
                message += f', {timer.tostring()}'
                pbar.set_description(message)
            except Exception as e:
                if str(e) == "element 0 of tensors does not require grad and does not have a grad_fn":
                    self.logger.error(f"Specific exception encountered at iteration {self.iteration}: {e}")
                    return
                else:
                    self.logger.error(f"Exception encountered at iteration {self.iteration}: {e}")
                    self.logger.error(traceback.format_exc())

                    # Extra diagnostics for CUDA OOM / huge allocations.
                    msg = str(e).lower()
                    if "out of memory" in msg or "cuda oom" in msg:
                        try:
                            alloc_mb = torch.cuda.memory_allocated() / (1024 ** 2)
                            reserv_mb = torch.cuda.memory_reserved() / (1024 ** 2)
                            max_alloc_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                            max_reserv_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
                            self.logger.error(
                                f"CUDA memory (MB) allocated={alloc_mb:.1f} reserved={reserv_mb:.1f} "
                                f"max_allocated={max_alloc_mb:.1f} max_reserved={max_reserv_mb:.1f}"
                            )
                        except Exception:
                            pass

                        # Log key tensor shapes to identify NxM blowups.
                        try:
                            if isinstance(data_dict, dict):
                                for k in ["points", "lengths", "ref_points", "src_points", "ref_lengths", "src_lengths"]:
                                    if k in data_dict and torch.is_tensor(data_dict[k]):
                                        self.logger.error(f"data_dict[{k}] shape={tuple(data_dict[k].shape)} dtype={data_dict[k].dtype} device={data_dict[k].device}")
                        except Exception:
                            pass

                        try:
                            self.logger.error(torch.cuda.memory_summary())
                        except Exception:
                            pass

                    # Prevent memory accumulation after failed iterations (especially after CUDA OOM).
                    try:
                        del output_dict
                    except Exception:
                        pass
                    try:
                        del result_dict
                    except Exception:
                        pass
                    try:
                        del data_dict
                    except Exception:
                        pass
                    torch.cuda.empty_cache()
                    continue
            finally:
                # Ensure no stale references keep CUDA tensors alive.
                try:
                    del output_dict
                except Exception:
                    pass
                try:
                    del result_dict
                except Exception:
                    pass
                try:
                    del data_dict
                except Exception:
                    pass

                # Adaptive cache trim to avoid VRAM saturation (prevents shared-memory spill on Windows).
                try:
                    props = torch.cuda.get_device_properties(0)
                    total = float(props.total_memory)
                    reserved = float(torch.cuda.memory_reserved())
                    if total > 0 and (reserved / total) >= self._vram_trim_ratio:
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            # torch.cuda.empty_cache()  # Extremely slow! Do not call periodically inside the inference loop
        self.after_test_epoch()
        summary_dict = summary_board.summary()
        message = get_log_string(result_dict=summary_dict, timer=timer)
        self.logger.critical(message)
