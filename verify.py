"""Verify data partitions and independently recompute every reported metric."""
import csv
import json
import numpy as np
import torch
from data import ROOT, load_data, split_indices, normalized, gaussian
from models import make_model
from metrics import classification_metrics


def main():
    config = json.loads((ROOT / 'config.json').read_text())
    x, y, xt, yt = load_data()
    assert x.shape == (60000,28,28) and xt.shape == (10000,28,28)
    train, val = split_indices(y, config)
    assert len(set(train)) == 12000 and len(set(val)) == 3000
    assert not set(train).intersection(val)
    assert np.all(np.bincount(y[train], minlength=10) == 1200)
    assert np.all(np.bincount(y[val], minlength=10) == 300)
    split = np.load(ROOT / 'results/split_indices.npz')
    assert np.array_equal(train, split['train']) and np.array_equal(val,split['validation'])
    predictions = np.load(ROOT / 'results/predictions.npz')
    assert np.array_equal(predictions['labels'], yt)
    with (ROOT / 'results/metrics.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4 * 3 * 7
    for row in rows:
        pred = predictions[f"{row['condition']}_{row['seed']}_{row['corruption']}"]
        assert pred.shape == yt.shape and np.all((pred>=0)&(pred<10))
        computed = classification_metrics(yt,pred)
        for key in ['accuracy','macro_f1','correct','n']:
            assert abs(float(row[key])-computed[key])<1e-10
    # Explicit hand-check of macro F1 and absent-class behavior.
    m = classification_metrics(np.array([0,0,1,1]), np.array([0,1,1,1]))
    assert abs(m['accuracy']-.75)<1e-12
    assert abs(m['macro_f1']-((2/3+4/5)/10))<1e-12
    for name, count in [('linear',7850),('cnn',105866)]:
        assert sum(p.numel() for p in make_model(name).parameters()) == count
    for name in config['conditions']:
        for seed in config['seeds']:
            checkpoint = torch.load(ROOT/'checkpoints'/f'{name}_{seed}.pt', weights_only=True)
            assert checkpoint['config'] == config
            with (ROOT/f'results/history_{name}_{seed}.csv').open() as f:
                history = list(csv.DictReader(f))
            assert len(history) == config['epochs']
            best = max(history, key=lambda r: float(r['selection_score']))
            assert checkpoint['best_epoch'] == int(best['epoch'])
    # Re-run inference from a saved checkpoint to rule out stale prediction files.
    torch.set_num_threads(4)
    model = make_model('cnn')
    checkpoint = torch.load(ROOT/'checkpoints/cnn_noise030_11.pt', weights_only=True)
    model.load_state_dict(checkpoint['state_dict']); model.eval()
    test_x = normalized(xt)
    noise = np.random.default_rng(config['test_noise_seed']).standard_normal(test_x.shape,dtype=np.float32)
    noisy_x = gaussian(test_x,.3,noise)
    with torch.inference_mode():
        fresh = model(torch.from_numpy(noisy_x[:512])).argmax(1).numpy()
    assert np.array_equal(fresh,predictions['cnn_noise030_11_gaussian_0.3'][:512])
    # Byte-identical pixel duplicates are counted, separately from index leakage.
    train_set = {image.tobytes() for image in x[train]}
    val_duplicates = sum(image.tobytes() in train_set for image in x[val])
    test_duplicates = sum(image.tobytes() in train_set for image in xt)
    receipt = {'status':'passed','metric_rows_recomputed':len(rows),'checkpoints_verified':12,
               'fresh_checkpoint_predictions_verified':512,
               'train_validation_index_overlap':0,'train_per_class':1200,'validation_per_class':300,
               'validation_images_identical_to_train':val_duplicates,
               'test_images_identical_to_train':test_duplicates,
               'test_class_counts':np.bincount(yt,minlength=10).tolist()}
    (ROOT / 'results/verification.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
