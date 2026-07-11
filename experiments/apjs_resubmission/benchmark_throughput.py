#!/usr/bin/env python3
"""Measure PI and CQT-DeiT per-pair preprocessing and GPU inference throughput."""

import json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import matplotlib, numpy as np, torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/classifier')); sys.path.insert(0,str(ROOT/'src/semd')); sys.path.insert(0,str(ROOT/'experiments/apjs_resubmission'))
from data_classifier import pad_or_trim
from model_classifier_ablation import BinaryPeriodicResNet1D_Ablation
from model import get_deit_tiny_distilled_enhanced
from prepare_cqt_cache_0228 import spectrum

DATA=Path('/root/autodl-tmp/qkzhang/SIS_data_0228'); N=256; BATCH=256; DEVICE=torch.device('cuda:0')
a=np.load(DATA/'SIS_data_strain_1.npy',mmap_mode='r'); b=np.load(DATA/'SIS_data_strain_2.npy',mmap_mode='r')

def pi_pre(i):
    out=[]
    for x in (a[i],b[i]):
        x=pad_or_trim(x,8192,2); x=(x-x.mean(axis=-1,keepdims=True))/(x.std(axis=-1,keepdims=True)+1e-8); out.append(x)
    return np.concatenate(out).astype('float32')
def cqt_pre(i):
    m=np.concatenate([spectrum(a[i]),spectrum(b[i])]); m=(m-m.min())/(m.max()-m.min())
    rgb=matplotlib.colormaps['viridis'](m,bytes=True)[...,:3].astype('float32').transpose(2,0,1)/255
    return (rgb-np.array([.485,.456,.406])[:,None,None])/np.array([.229,.224,.225])[:,None,None]
def timed(fn,workers):
    start=time.perf_counter()
    if workers==1: values=[fn(i) for i in range(N)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool: values=list(pool.map(fn,range(N)))
    elapsed=time.perf_counter()-start
    return np.stack(values).astype('float32'),elapsed
pi_values,pi_serial=timed(pi_pre,1); _,pi_parallel=timed(pi_pre,16)
cqt_values,cqt_serial=timed(cqt_pre,1); _,cqt_parallel=timed(cqt_pre,16)

pi=BinaryPeriodicResNet1D_Ablation(in_channels=1,d_model=256,width_scale=4.0,use_snake=False,use_se=True,use_physics_fusion=True).to(DEVICE).eval()
pi.load_state_dict(torch.load(ROOT/'runs/apjs_resubmission_final_v1/pi_resnet_sis_noisy_seed42/best.pt',map_location=DEVICE,weights_only=False)['model_state_dict'])
cqt=get_deit_tiny_distilled_enhanced(num_classes=2,pretrained=False,hidden_dim=512,dropout_rate=.5,freeze_backbone=False).to(DEVICE).eval()
cqt.load_state_dict(torch.load(ROOT/'runs/apjs_resubmission_final_v1/cqt_deit_sis_noisy_seed42/best.pth',map_location=DEVICE,weights_only=False)['model_state_dict'])
def gpu_time(model,values):
    x=torch.from_numpy(values).to(DEVICE); repetitions=100
    with torch.no_grad():
        for _ in range(10): model(x)
        torch.cuda.synchronize(); start=time.perf_counter()
        for _ in range(repetitions): model(x)
        torch.cuda.synchronize(); elapsed=time.perf_counter()-start
    return elapsed/(repetitions*len(values))
pi_gpu=gpu_time(pi,pi_values); cqt_gpu=gpu_time(cqt,cqt_values)
result={"status":"complete","hardware":{"gpu":torch.cuda.get_device_name(DEVICE),"cpu_threads":16},"pairs":N,"batch_size":BATCH,
"pi_resnet":{"preprocess_ms_per_pair_serial":1000*pi_serial/N,"preprocess_ms_per_pair_16_threads_wall":1000*pi_parallel/N,"gpu_inference_ms_per_pair":1000*pi_gpu,"gpu_pairs_per_second":1/pi_gpu},
"cqt_deit":{"preprocess_ms_per_pair_serial_including_cqt":1000*cqt_serial/N,"preprocess_ms_per_pair_16_threads_wall_including_cqt":1000*cqt_parallel/N,"gpu_inference_ms_per_pair":1000*cqt_gpu,"gpu_pairs_per_second":1/cqt_gpu},
"note":"Preprocessing wall-time per pair is total wall time divided by N; the 16-thread figure reflects pipeline throughput, not single-pair latency."}
out=ROOT/'runs/apjs_resubmission_final_v1/throughput';out.mkdir(parents=True,exist_ok=True);(out/'throughput.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
