"""CPU-only materialization of absent frozen branches, without training or cache writes."""

import json
import time
from pathlib import Path

from .phase3_replay import calibration_populations, json_read, parents
from .protocol import cache_key, digest, read_records


def main(ctx, config):
    import torch
    from .encoders import FrozenEncoders
    from .experiment import _forward, verify_resources
    from .model import SaverCore
    from .protocol import FeatureStore

    if torch.cuda.is_available() or config.get('device', 'cpu') != 'cpu':
        raise RuntimeError('Phase3 CPU extraction requires CUDA_VISIBLE_DEVICES empty')
    torch.set_num_threads(4)
    root = Path(config['phase1_root'])
    source_parents = parents(root, ctx.seed)
    saver = source_parents['saver']
    source = json_read(saver / 'config.json')
    source['manifest'] = str(Path(config['source_repo']) / source['manifest'])
    source['cache'] = str(Path(config['source_repo']) / source['cache'])
    identity = json_read(saver / 'data_identity.json')
    records = read_records(source['manifest'])
    encoder_identity = verify_resources(source, 'mre')
    populations = calibration_populations(records, identity, ctx.seed)
    original_ids = set(identity['split_ids']['calibration'])
    extension = [row for row in populations[.2] if row['id'] not in original_ids]
    dev = sorted([row for row in records if row['split'] == 'dev'], key=lambda row: row['id'])
    ctx.tags.update(stage='phase3', study='R2_extract', method='saver', device='cpu')
    ctx.save_json('input_identity.json', {'parents': {key: str(value) for key, value in source_parents.items()},
                  'checkpoint_sha256': {key: digest(value / 'best.pt') for key, value in source_parents.items()},
                  'manifest_sha256': digest(source['manifest']), 'test_labels_opened': False,
                  'source_repo': config['source_repo'], 'resource_identity': encoder_identity,
                  'calibration_sizes': {str(key): len(value) for key, value in populations.items()}})
    encoders = FrozenEncoders(source['text_model'], source['vision_model'], device='cpu')

    class ReadOnlyStore(FeatureStore):
        def __init__(self):
            super().__init__(source['cache'], encoders, identity=encoder_identity)
            self.hits = self.misses = self.detail_hits = self.detail_misses = 0
            self.memory = {}

        def text_global(self, record, task, visual=True):
            path = self.root / (cache_key(record) + '.pt')
            if path.exists():
                payload = torch.load(path, map_location='cpu', weights_only=True)
                if not visual or payload.get('global') is not None:
                    self.hits += 1
                    return payload
            self.misses += 1
            units = encoders.encode_text(record, task, return_units=False)
            return {'units': {key: value.detach().cpu() for key, value in units.items()
                              if isinstance(value, torch.Tensor)}, 'truncated': units['truncated'],
                    'global': encoders.encode_global(record['images']).cpu() if visual else None}

        def detail(self, image, cache=True):
            import hashlib
            path = self.root / ('detail-' + hashlib.sha256(str(image).encode()).hexdigest() + '.pt')
            if path.exists():
                self.detail_hits += 1
                return torch.load(path, map_location='cpu', weights_only=True)
            if image in self.memory:
                return self.memory[image]
            self.detail_misses += 1
            patches, mask = encoders.encode_detail(image, return_mask=True)
            result = patches[mask.bool()]
            if len(self.memory) < 256:
                self.memory[image] = result
            return result

    store = ReadOnlyStore()
    steps = [('saver_dev', dev, 'saver'), ('calibration_extension', extension, 'saver'),
             ('independent_dev', dev, 'text_only')]
    for name, population, method in steps:
        parent = source_parents[method]
        model = SaverCore(len(identity['labels']), text_dim=encoders.text_hidden, vision_dim=encoders.vision_hidden)
        model.load_state_dict(torch.load(parent / 'best.pt', map_location='cpu', weights_only=True))
        model.eval().requires_grad_(False)
        start = time.monotonic()
        with torch.inference_mode():
            for index, record in enumerate(population):
                output, _, labels = _forward(model, store, record, 'mre', identity['labels'], method, 1000,
                                             force_details=method == 'saver', cache_detail=True)
                row = {'id': record['id'], 'gold': labels.tolist(),
                       'gate': output['gate'].tolist(), 'eligible': output['image_available'].tolist(),
                       'text_prediction': output['text_logits'].argmax(-1).tolist(),
                       'vision_prediction': output['vision_logits'].argmax(-1).tolist(),
                       'prediction': output['logits'].argmax(-1).tolist(), 'selected': output['selected'].tolist()}
                if method == 'saver':
                    row['unit_ce_gain'] = (output['text_losses'] - output['vision_losses']).tolist()
                ctx.write_rows(name, [row])
                if index % 100 == 0:
                    ctx.log_event(kind='frozen_cpu_progress', population=name, completed=index + 1,
                                  total=len(population), elapsed_seconds=time.monotonic() - start)
        ctx.set_metrics({name + '_rows': len(population), name + '_seconds': time.monotonic() - start})
        del model
    ctx.save_json('inference_scope.json', {'device': 'cpu', 'gpu_hours': 0., 'training': False,
                  'read_only_source_cache': str(store.root), 'cache_hits': store.hits, 'cache_misses': store.misses,
                  'detail_cache_hits': store.detail_hits, 'detail_cache_misses': store.detail_misses,
                  'new_feature_cache_bytes': 0, 'precision': 'float32 task head; historical cached backbone tensors retained; absent backbone tensors recomputed CPU float32',
                  'latency_claim': 'CPU extraction wall time is not a T4 inference latency measurement'})
    ctx.set_metrics({'training_updates': 0, 'gpu_hours_actual': 0., 'source_cache_writes': 0,
                     'dev_posts_complete': True, 'calibration_extension_rows': len(extension)})
