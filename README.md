import os

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

BASE = "/nfsshare/users/manjula/svamitva_project"

csv_file = f"{BASE}/final_analysis_summary.csv"

if not os.path.exists(csv_file):

    print("final_analysis_summary.csv not found")
else:

    df = pd.read_csv(csv_file)

"""
SVAMITVA Project — Complete Pipeline
=====================================
Stage 1  Mask generation  ->  data/training/masks_raster/

Stage 2  Patch extraction ->  data/training/patches/{images,masks}/

Stage 3  Augmentation     ->  same patch dirs (minority classes)

Stage 4  Train DuSA U-Net ->  outputs/best_model.pth

Stage 5  Train R-CNN      ->  outputs/final/faster_rcnn_utilities.pth

Stage 6  Inference        ->  outputs/predictions/ + outputs/rcnn_utilities/

Stage 7  Merge+Eval+Vis   ->  outputs/final_predictions/ + charts

Usage
-----
python svamitva_pipeline.py --stage all      # full run

python svamitva_pipeline.py --stage 1        # masks only

python svamitva_pipeline.py --stage 1,2,3    # comma list

python svamitva_pipeline.py --stage 4,5      # training only
"""

# stdlib
import argparse, glob, os, random, sys, warnings

from collections import defaultdict

warnings.filterwarnings("ignore")

random.seed(42)

HPC_PROJECT = "/nfsshare/users/manjula/svamitva_project"

if os.path.isdir(HPC_PROJECT):

    sys.path.insert(0, HPC_PROJECT)

# ============================================================
# SHARED CONSTANTS
# ============================================================

CLASS_NAMES = {
    0: "Background", 1: "RCC_Roof",  2: "Tiled_Roof",  3: "Tin_Roof",
    4: "Thatched_Roof", 5: "Road",   6: "Waterbody",   7: "Transformer",
    8: "Tank",          9: "Well",
}

MODEL_SOURCES = {
    "RCC_Roof":"DuSA U-Net","Tiled_Roof":"DuSA U-Net","Tin_Roof":"DuSA U-Net",
    "Thatched_Roof":"DuSA U-Net","Road":"DuSA U-Net","Waterbody":"DuSA U-Net",
    "Transformer":"Faster R-CNN","Tank":"Faster R-CNN","Well":"DuSA U-Net",
}

FEATURE_COLORS = {
    1:{"name":"RCC_Roof",      "hex":"#E74C3C","rgb":(231, 76, 60)},
    2:{"name":"Tiled_Roof",    "hex":"#E67E22","rgb":(230,126, 34)},
    3:{"name":"Tin_Roof",      "hex":"#F1C40F","rgb":(241,196, 15)},
    4:{"name":"Thatched_Roof", "hex":"#8B4513","rgb":(139, 69, 19)},
    5:{"name":"Road",          "hex":"#95A5A6","rgb":(149,165,166)},
    6:{"name":"Waterbody",     "hex":"#3498DB","rgb":( 52,152,219)},
    7:{"name":"Transformer",   "hex":"#9B59B6","rgb":(155, 89,182)},
    8:{"name":"Tank",          "hex":"#1ABC9C","rgb":( 26,188,156)},
    9:{"name":"Well",          "hex":"#2ECC71","rgb":( 46,204,113)},
}

WORKING_VILLAGES = ["BADETUMNAR","MURDANDA","NAGUL","28996","PINDORI","TIMMOWAL"]
HPC_PROJECT = "/nfsshare/users/manjula/svamitva_project"
DATA_ROOT   = f"{HPC_PROJECT}/data"

PATHS = {
    "train_images": f"{DATA_ROOT}/training/images",
    "train_masks":  f"{DATA_ROOT}/training/masks_raster",
    "patch_images": f"{DATA_ROOT}/training/patches/images",
    "patch_masks":  f"{DATA_ROOT}/training/patches/masks",
    "test_images":  f"{DATA_ROOT}/testing/images",
    "unet_out":   f"{HPC_PROJECT}/outputs/predictions",
    "rcnn_out":   f"{HPC_PROJECT}/outputs/rcnn_utilities",
    "final_out":  f"{HPC_PROJECT}/outputs/final_predictions",
    "unet_model": f"{HPC_PROJECT}/outputs/best_model.pth",
    "rcnn_model": f"{HPC_PROJECT}/outputs/final/faster_rcnn_utilities.pth",
    "samples":    f"{HPC_PROJECT}/samples",
}


_SHP_CANDIDATES = [
    "/nfsshare/users/manjula/svamitva_project/data/training/masks",
    "/nfsshare/users/manjula/svamitva_project/data/raw/shp-file",
    "/nfsshare/users/manjula/svamitva_project/data/raw/PB_training_dataSet_shp_file",
]

def _banner(t): print("\n"+"="*70+"\n"+t+"\n"+"="*70)
def _mkdirs(*d): [os.makedirs(x, exist_ok=True) for x in d]
def _find_shp():
    for d in _SHP_CANDIDATES:
        if os.path.isdir(d): return d
    return None

# ============================================================
# STAGE 1 — MASK GENERATION
# ============================================================

def stage1_mask_generation():
    _banner("STAGE 1 — MASK GENERATION")
    import numpy as np, rasterio
    from rasterio import features as rf
    import geopandas as gpd
    from shapely.geometry import box as sbox

    shp_dir = _find_shp()
    if not shp_dir:
        print("ERROR: shapefile dir not found. Checked:", _SHP_CANDIDATES); return False

    tifs = sorted(glob.glob(os.path.join(PATHS["train_images"],"*.tif")))
    if not tifs:
        print("ERROR: no .tif in", PATHS["train_images"]); return False

    print(f"  Shapefiles : {shp_dir}")
    print(f"  Images     : {len(tifs)}")

    # load shapefiles once
    layers = {}
    for fname, cid in [
    ("Built_Up_Area_type.shp", None),
    ("Built_Up_Area_typ.shp", None),
    ("Road.shp", 5),
    ("Water_Body.shp", 6),
    ("Utility.shp", 7)
]:
        p = os.path.join(shp_dir, fname)
        if os.path.exists(p):
            layers[fname] = (gpd.read_file(p), cid)
            print(f"  loaded {fname} ({len(layers[fname][0])} rows)")
        else:
            print(f"  WARNING: {fname} not found")

    _mkdirs(PATHS["train_masks"])
    ok = 0
    for tif in tifs:
        stem = os.path.splitext(os.path.basename(tif))[0]
        out  = os.path.join(PATHS["train_masks"], f"{stem}_mask.tif")
        if os.path.exists(out):
            print(f"  {stem}: already exists"); ok+=1; continue
        try:
            with rasterio.open(tif) as src:
                H,W = src.height, src.width
                tfm = src.transform
                meta = src.meta.copy()
                bounds = sbox(*src.bounds)
            mask = np.zeros((H,W), dtype=np.uint8)
            for fname,(gdf,cid) in layers.items():
                gdf_f = gdf[gdf.intersects(bounds)]
                if gdf_f.empty: continue
                if fname=="Built_Up_Area_type.shp":
                    rm={1:1,2:2,3:3,4:4}
                    shapes=[(r.geometry,rm.get(int(r["Roof_type"]),1))
                            for _,r in gdf_f.iterrows() if r.geometry]
                else:
                    shapes=[(g,cid) for g in gdf_f.geometry if g]
                if shapes:
                    burned=rf.rasterize(shapes,out_shape=(H,W),
                                        transform=tfm,fill=0,dtype=np.uint8)
                    mask=np.maximum(mask,burned)
            meta.update(dtype=rasterio.uint8,count=1,compress="lzw",nodata=None)
            meta.pop("photometric",None)
            with rasterio.open(out,"w",**meta) as dst: dst.write(mask,1)
            print(f"  {stem}_mask.tif  fg={int(np.sum(mask>0)):,}")
            ok+=1
        except Exception as e:
            print(f"  ERROR {stem}: {e}")
    print(f"\n  Done: {ok}/{len(tifs)} masks -> {PATHS['train_masks']}/")
    return ok>0

# ============================================================
# STAGE 2 — PATCH EXTRACTION
# ============================================================

def stage2_patch_extraction():
    _banner("STAGE 2 — PATCH EXTRACTION")
    import numpy as np, rasterio
    from rasterio.windows import Window
    from PIL import Image

    PS,ST = 512,256
    _mkdirs(PATHS["patch_images"],PATHS["patch_masks"])

    all_masks = sorted(glob.glob(os.path.join(PATHS["train_masks"],"*.tif")))
    if not all_masks:
        print("ERROR: no masks in",PATHS["train_masks"],". Run Stage 1."); return False

    pairs=[]
    for mp in all_masks:
        stem = os.path.splitext(os.path.basename(mp))[0]
        img_stem = stem[:-5] if stem.endswith("_mask") else stem
        if not any(v.upper() in img_stem.upper() for v in WORKING_VILLAGES): continue
        ip = os.path.join(PATHS["train_images"], f"{img_stem}.tif")
        if not os.path.exists(ip):
            print(f"  WARNING: image not found for {os.path.basename(mp)}"); continue
        pairs.append((ip,mp,img_stem))

    if not pairs:
        print("ERROR: no village pairs found.")
        print("  Available masks:", [os.path.basename(m) for m in all_masks[:10]])
        return False

    total_new=0
    for img_path,mask_path,stem in pairs:
        exist=len(glob.glob(os.path.join(PATHS["patch_images"],f"{stem}_*.png")))
        new=skip=0
        try:
            with rasterio.open(img_path) as isrc, rasterio.open(mask_path) as msrc:
                H,W,nb = isrc.height, isrc.width, isrc.count
                print(f"  [{stem}] {W}x{H}  existing={exist}")
                for y in range(0,H-PS+1,ST):
                    for x in range(0,W-PS+1,ST):
                        pn = f"{stem}_{y}_{x}.png"
                        pi = os.path.join(PATHS["patch_images"],pn)
                        pm = os.path.join(PATHS["patch_masks"],pn)
                        if os.path.exists(pi): skip+=1; continue
                        win=Window(x,y,PS,PS)
                        ip=isrc.read(window=win); mp=msrc.read(1,window=win)
                        if mp.max()==0: skip+=1; continue
                        ip=ip[:3] if nb>=3 else np.repeat(ip,3,axis=0)
                        ip=np.transpose(ip,(1,2,0)).astype(np.float32)
                        lo,hi=ip.min(),ip.max()
                        ip=((ip-lo)/(hi-lo+1e-8)*255).astype(np.uint8)
                        Image.fromarray(ip).save(pi)
                        Image.fromarray(mp).save(pm)
                        new+=1
        except Exception as e:
            print(f"  ERROR {stem}: {e}"); continue
        print(f"    new={new} skipped={skip}")
        total_new+=new

    total=len(glob.glob(os.path.join(PATHS["patch_images"],"*.png")))
    print(f"\n  New={total_new}  Total on disk={total:,}")
    return total>0

# ============================================================
# STAGE 3 — AUGMENTATION
# ============================================================

def stage3_augmentation():
    _banner("STAGE 3 — MINORITY-CLASS AUGMENTATION")
    import numpy as np, cv2

    TARGETS={8:("Tank",5), 9:("Well",65)}
    idir=PATHS["patch_images"]; mdir=PATHS["patch_masks"]

    for cval,(label,n) in TARGETS.items():
        srcs=[]
        for f in sorted(os.listdir(idir)):
            if f.startswith("aug_"): continue
            mp=os.path.join(mdir,f)
            if not os.path.exists(mp): continue
            m=cv2.imread(mp,cv2.IMREAD_GRAYSCALE)
            if m is not None and cval in np.unique(m): srcs.append(f)
        print(f"\n  {label}: {len(srcs)} sources x {n} = {len(srcs)*n} augments")
        if not srcs: print("  WARNING: no source patches"); continue
        cnt=0
        for f in srcs:
            img =cv2.imread(os.path.join(idir,f))
            mask=cv2.imread(os.path.join(mdir,f),cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None: continue
            for _ in range(n):
                fl=random.choice([-2,-1,0,1])
                ai=cv2.flip(img,fl)  if fl!=-2 else img.copy()
                am=cv2.flip(mask,fl) if fl!=-2 else mask.copy()
                ai=cv2.convertScaleAbs(ai,alpha=random.uniform(0.7,1.4),
                                          beta=random.uniform(-40,40))
                nf=f"aug_{label.lower()}_{cnt}_{f}"
                cv2.imwrite(os.path.join(idir,nf),ai)
                cv2.imwrite(os.path.join(mdir,nf),am)
                cnt+=1
        print(f"  Generated {cnt}")

    total=len(glob.glob(os.path.join(idir,"*.png")))
    print(f"\n  Total patches after augmentation: {total:,}")
    return True

# ============================================================
# STAGE 4 — TRAIN DuSA U-Net
# ============================================================

def stage4_train_unet():
    _banner("STAGE 4 — TRAIN DuSA U-Net")
    import numpy as np, torch, torch.nn as nn, torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, Subset
    from PIL import Image
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from models.dusa_unet import DuSA_UNet

    os.environ["CUDA_VISIBLE_DEVICES"]="5"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
    torch.manual_seed(42); np.random.seed(42)

    AUG_TR=A.Compose([
        A.HorizontalFlip(p=0.5),A.VerticalFlip(p=0.5),A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.3),A.RandomGamma(p=0.3),A.GaussNoise(p=0.2),
        A.ElasticTransform(p=0.2,alpha=1,sigma=50),A.CLAHE(p=0.3),
        A.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),ToTensorV2()])
    AUG_VA=A.Compose([
        A.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),ToTensorV2()])

    class PDS(Dataset):
        def __init__(self,aug):
            self.imgs =sorted(glob.glob(os.path.join(PATHS["patch_images"],"*.png")))
            self.masks=sorted(glob.glob(os.path.join(PATHS["patch_masks"], "*.png")))
            self.aug=aug
        def __len__(self): return len(self.imgs)
        def __getitem__(self,i):
            img =np.array(Image.open(self.imgs[i]).convert("RGB"))
            mask=np.clip(np.array(Image.open(self.masks[i])),0,9)
            out =self.aug(image=img,mask=mask)
            return out["image"],out["mask"].long()

    n=len(glob.glob(os.path.join(PATHS["patch_images"],"*.png")))
    if n==0:
        print("ERROR: no patches. Run Stages 1-3 first."); return False

    idx=list(range(n)); random.shuffle(idx)
    vn=int(0.2*n); tr_i=idx[vn:]; va_i=idx[:vn]
    tr_ds=Subset(PDS(AUG_TR),tr_i); va_ds=Subset(PDS(AUG_VA),va_i)
    tr_ld=DataLoader(tr_ds,batch_size=16,shuffle=True, num_workers=8,pin_memory=True,persistent_workers=True)
    va_ld=DataLoader(va_ds,batch_size=16,shuffle=False,num_workers=8,pin_memory=True,persistent_workers=True)
    print(f"  Train={len(tr_ds):,}  Val={len(va_ds):,}")

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=DuSA_UNet(n_classes=10).to(device)
    print(f"  Device={device}  Params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    start,best=0,0.0
    for ck in [PATHS["unet_model"],"outputs/best_model_resumed.pth"]:
        if os.path.exists(ck):
            model.load_state_dict(torch.load(ck,map_location=device))
            start=12; best=91.75
            print(f"  Resumed: {ck}"); break

    class LSCe(nn.Module):
        def __init__(self,s=0.1): super().__init__(); self.s=s
        def forward(self,p,t):
            n=p.size(1); lp=nn.functional.log_softmax(p,dim=1)
            with torch.no_grad():
                td=torch.full_like(p,self.s/(n-1)); td.scatter_(1,t.unsqueeze(1),1-self.s)
            return (-td*lp).sum(1).mean()

    crit=LSCe(0.1)
    opt =optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    sch =optim.lr_scheduler.OneCycleLR(opt,max_lr=1e-3,total_steps=len(tr_ld)*50,
         pct_start=0.3,anneal_strategy="cos",
         last_epoch=(start*len(tr_ld)-1) if start else -1)
    scaler=torch.cuda.amp.GradScaler()
    _mkdirs("outputs")
    print(f"  Starting epoch {start+1}/50  best={best:.2f}%\n")

    for epoch in range(start,50):
        model.train(); tl=0.0
        for imgs,masks in tr_ld:
            imgs,masks=imgs.to(device),masks.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(): loss=crit(model(imgs),masks)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            sch.step(); tl+=loss.item()
        tl/=len(tr_ld)
        model.eval(); correct=total=0; vl=0.0
        with torch.no_grad():
            for imgs,masks in va_ld:
                imgs,masks=imgs.to(device),masks.to(device)
                with torch.cuda.amp.autocast(): out=model(imgs); loss=crit(out,masks)
                vl+=loss.item(); pred=torch.argmax(out,1)
                correct+=(pred==masks).sum().item(); total+=masks.numel()
        vl/=len(va_ld); acc=correct/total*100
        flag=""
        if acc>best:
            best=acc; torch.save(model.state_dict(),PATHS["unet_model"]); flag="  <- best"
        print(f"  Ep {epoch+1:3d}/50  tr={tl:.4f}  val={vl:.4f}  acc={acc:.2f}%{flag}")
        torch.cuda.empty_cache()

    print(f"\n  Best accuracy: {best:.2f}%  saved -> {PATHS['unet_model']}")
    return True

# ============================================================
# STAGE 5 — TRAIN Faster R-CNN
# ============================================================

def stage5_train_rcnn():
    _banner("STAGE 5 — TRAIN Faster R-CNN (Utilities)")
    import math, numpy as np, torch, cv2
    from torch.utils.data import Dataset, DataLoader, Subset
    from torchvision import transforms as T
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from tqdm import tqdm

    os.environ["CUDA_VISIBLE_DEVICES"]="6"
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    KERNEL=np.ones((17,17),np.uint8)
    M2C={7:1,8:2,9:3}

    class UDS(Dataset):
        def __init__(self):
            self.imgs=sorted(os.listdir(PATHS["patch_images"]))
        def __len__(self): return len(self.imgs)
        def __getitem__(self,idx):
            f=self.imgs[idx]
            img=cv2.cvtColor(cv2.imread(os.path.join(PATHS["patch_images"],f)),
                              cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
            mask=cv2.imread(os.path.join(PATHS["patch_masks"],f),cv2.IMREAD_GRAYSCALE)
            boxes,labels=[],[]
            for mv,ci in M2C.items():
                bm=cv2.dilate((mask==mv).astype(np.uint8),KERNEL,iterations=1)
                try: nl,_,stats,_=cv2.connectedComponentsWithStats(bm,connectivity=8)
                except cv2.error: continue
                for i in range(1,nl):
                    x,y=stats[i,cv2.CC_STAT_LEFT],stats[i,cv2.CC_STAT_TOP]
                    w,h=stats[i,cv2.CC_STAT_WIDTH],stats[i,cv2.CC_STAT_HEIGHT]
                    if w>=2 and h>=2:
                        boxes.append([float(x),float(y),float(x+w),float(y+h)]); labels.append(ci)
            boxes=torch.as_tensor(boxes,dtype=torch.float32)
            labels=torch.as_tensor(labels,dtype=torch.int64); iid=torch.tensor([idx])
            if len(boxes):
                tgt={"boxes":boxes,"labels":labels,"image_id":iid,
                     "area":(boxes[:,3]-boxes[:,1])*(boxes[:,2]-boxes[:,0]),
                     "iscrowd":torch.zeros(len(boxes),dtype=torch.int64)}
            else:
                tgt={"boxes":torch.zeros((0,4),dtype=torch.float32),
                     "labels":torch.zeros((0,),dtype=torch.int64),"image_id":iid,
                     "area":torch.zeros((0,),dtype=torch.float32),
                     "iscrowd":torch.zeros((0,),dtype=torch.int64)}
            return T.ToTensor()(img),tgt

    def colfn(b): return tuple(zip(*b))

    def get_m(nc=4):
        anc=AnchorGenerator(sizes=((16,),(32,),(64,),(128,),(256,)),
                            aspect_ratios=((0.5,1.0,2.0),)*5)
        m=fasterrcnn_resnet50_fpn(weights="DEFAULT",rpn_anchor_generator=anc,box_nms_thresh=0.2)
        inf=m.roi_heads.box_predictor.cls_score.in_features
        m.roi_heads.box_predictor=FastRCNNPredictor(inf,nc); return m

    full=UDS()
    masks_list=sorted(os.listdir(PATHS["patch_masks"]))
    print("  Scanning utility patches...")
    ui,ei=[],[]
    for i,f in enumerate(tqdm(masks_list,desc="scan")):
        m=cv2.imread(os.path.join(PATHS["patch_masks"],f),cv2.IMREAD_GRAYSCALE)
        if m is None: continue
        has=False
        if np.any(np.isin(m,[7,8,9])):
            for mv in [7,8,9]:
                bm=cv2.dilate((m==mv).astype(np.uint8),KERNEL,iterations=1)
                try:
                    nl,_,stats,_=cv2.connectedComponentsWithStats(bm,connectivity=8)
                    for j in range(1,nl):
                        if stats[j,cv2.CC_STAT_WIDTH]>=2 and stats[j,cv2.CC_STAT_HEIGHT]>=2:
                            has=True; break
                except cv2.error: pass
                if has: break
        (ui if has else ei).append(i)

    ne=int(len(ui)*0.20); random.shuffle(ei); active=ui+ei[:ne]
    print(f"  Utility={len(ui)}  Hard-neg={ne}")
    sub=Subset(full,active); tn=int(0.8*len(sub))
    gen=torch.Generator().manual_seed(42)
    tr,va=torch.utils.data.random_split(sub,[tn,len(sub)-tn],generator=gen)
    tr_ld=DataLoader(tr,batch_size=2,shuffle=True, num_workers=2,collate_fn=colfn)
    va_ld=DataLoader(va,batch_size=1,shuffle=False,num_workers=2,collate_fn=colfn)
    print(f"  Train={len(tr)}  Val={len(va)}")

    model=get_m(4).to(device)
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.SGD(params,lr=5e-4,momentum=0.9,weight_decay=5e-4)
    sch=torch.optim.lr_scheduler.StepLR(opt,step_size=10,gamma=0.2)
    _mkdirs("outputs/final")

    def cdist(b1,b2):
        return math.hypot((b1[0]+b1[2])/2-(b2[0]+b2[2])/2,(b1[1]+b1[3])/2-(b2[1]+b2[3])/2)

    for epoch in range(40):
        model.train(); el=0.0
        for imgs,tgts in tr_ld:
            imgs=[i.to(device) for i in imgs]
            tgts=[{k:v.to(device) for k,v in t.items()} for t in tgts]
            ld=model(imgs,tgts); loss=sum(ld.values())
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.0)
            opt.step(); el+=loss.item()
        sch.step()
        model.eval(); cc={1:0,2:0,3:0}; ct={1:0,2:0,3:0}; cp={1:0,2:0,3:0}
        with torch.no_grad():
            for vi,vt in va_ld:
                vi=[i.to(device) for i in vi]
                vt=[{k:v.to(device) for k,v in t.items()} for t in vt]
                out=model(vi)
                for i,o in enumerate(out):
                    gb=vt[i]["boxes"].cpu().numpy(); gl=vt[i]["labels"].cpu().numpy()
                    pb=o["boxes"].cpu().numpy(); pl=o["labels"].cpu().numpy()
                    ps=o["scores"].cpu().numpy(); keep=ps>=0.50; pb=pb[keep]; pl=pl[keep]
                    for cls in [1,2,3]:
                        gb_c=gb[gl==cls]; pb_c=pb[pl==cls]
                        ct[cls]+=len(gb_c); cp[cls]+=len(pb_c)
                        mg=set()
                        for pb_i in pb_c:
                            bd,bi=1e9,-1
                            for gi,gb_i in enumerate(gb_c):
                                if gi in mg: continue
                                d=cdist(pb_i,gb_i)
                                if d<bd: bd,bi=d,gi
                            if bd<=10.0: cc[cls]+=1; mg.add(bi)
        tc=sum(cc.values()); tg=sum(ct.values()); tp_=sum(cp.values())
        pr=tc/tp_ if tp_ else 0; rc=tc/tg if tg else 0
        print(f"  Ep {epoch+1:2d}/40  loss={el/len(tr_ld):.4f}  prec={pr:.2%}  recall={rc:.2%}")
        torch.cuda.empty_cache()

    torch.save(model.state_dict(),PATHS["rcnn_model"])
    print(f"  Saved -> {PATHS['rcnn_model']}")
    return True

# ============================================================
# STAGE 6 — INFERENCE
# ============================================================

def stage6_inference():
    _banner("STAGE 6 — INFERENCE ON TESTING VILLAGES")
    import numpy as np, torch, rasterio
    from rasterio import features as rf
    from rasterio.windows import Window
    import geopandas as gpd
    from shapely.geometry import shape, Point
    import cv2
    from models.dusa_unet import DuSA_UNet
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.anchor_utils import AnchorGenerator

    os.environ["CUDA_VISIBLE_DEVICES"]="6"
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_imgs=sorted(glob.glob(os.path.join(PATHS["test_images"],"*.tif")))
    if not test_imgs:
        print("ERROR: no test images in",PATHS["test_images"]); return False
    print(f"  Test villages: {len(test_imgs)}")

    THR={1:0.45,2:0.40,3:0.50,4:0.35,5:0.55,6:0.50,7:0.30,8:0.30,9:0.30}

    def tta(model,p):
        ps=[]
        with torch.no_grad():
            ps.append(model(p))
            for dims in ([3,],[2,],[2,3]):
                fp=torch.flip(p,dims=dims); o=model(fp)
                for d in reversed(dims): o=torch.flip(o,dims=[d])
                ps.append(o)
        return torch.stack(ps).mean(0)

    def athresh(sm):
        v=torch.zeros_like(sm)
        for c,t in THR.items(): v[c]=(sm[c]>t).float()
        return torch.argmax(v,0)

    def mclean(m,ks=3):
        k=np.ones((ks,ks),np.uint8)
        m=cv2.morphologyEx(m,cv2.MORPH_OPEN,k); m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,k)
        cnts,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c)<100: cv2.drawContours(m,[c],-1,0,-1)
        return m

    if not os.path.exists(PATHS["unet_model"]):
        print("ERROR: U-Net not found. Run Stage 4."); return False

    unet=DuSA_UNet(n_classes=10).to(device)
    unet.load_state_dict(torch.load(PATHS["unet_model"],map_location=device))
    unet.eval()
    _mkdirs(PATHS["unet_out"])
    FC={k:v for k,v in CLASS_NAMES.items() if k>0}
    PS,ST=512,256

    print("\n  [6a] U-Net segmentation...")
    for ip in test_imgs:
        base=os.path.splitext(os.path.basename(ip))[0]
        out=os.path.join(PATHS["unet_out"],f"{base}_features.gpkg")
        print(f"    {base}...",end=" ",flush=True)
        try:
            with rasterio.open(ip) as src:
                H,W=src.height,src.width; tfm,crs=src.transform,src.crs; nb=src.count
                pf=np.zeros((H,W),dtype=np.uint8)
                for y in range(0,H-PS+1,ST):
                    for x in range(0,W-PS+1,ST):
                        p=src.read(window=Window(x,y,PS,PS))
                        if nb==4: p=p[:3]
                        elif nb==1: p=np.repeat(p,3,axis=0)
                        pt=torch.FloatTensor(p).unsqueeze(0).to(device)
                        sm=tta(unet,pt); pr=athresh(sm[0])
                        pc=mclean(pr.cpu().numpy().astype(np.uint8))
                        pf[y:y+PS,x:x+PS]=np.maximum(pf[y:y+PS,x:x+PS],pc)
            feats=[]
            for cid,cn in FC.items():
                cm=(pf==cid).astype(np.uint8)
                if not cm.any(): continue
                for g,v in rf.shapes(cm,transform=tfm):
                    if v==1:
                        gs=shape(g); feats.append({"geometry":gs,"class_id":cid,"class_name":cn,"area":gs.area})
            if feats: gpd.GeoDataFrame(feats,crs=crs).to_file(out,driver="GPKG")
            print(f"{len(feats):,}")
        except Exception as e:
            print(f"ERROR: {e}")
        torch.cuda.empty_cache()

    if not os.path.exists(PATHS["rcnn_model"]):
        print("\n  WARNING: R-CNN model not found, skipping 6b")
    else:
        def get_r(nc=4):
            anc=AnchorGenerator(sizes=((16,),(32,),(64,),(128,),(256,)),aspect_ratios=((0.5,1.0,2.0),)*5)
            m=fasterrcnn_resnet50_fpn(weights="DEFAULT",rpn_anchor_generator=anc,box_nms_thresh=0.2)
            inf=m.roi_heads.box_predictor.cls_score.in_features
            m.roi_heads.box_predictor=FastRCNNPredictor(inf,nc); return m

        rcnn=get_r(4).to(device)
        rcnn.load_state_dict(torch.load(PATHS["rcnn_model"],map_location=device))
        rcnn.eval(); _mkdirs(PATHS["rcnn_out"])
        RC={1:"Transformer",2:"Tank",3:"Well"}; RP,RS=1024,512
        import pandas as pd; summary=[]
        print("\n  [6b] R-CNN utility detection...")
        for ip in test_imgs:
            base=os.path.splitext(os.path.basename(ip))[0]
            out=os.path.join(PATHS["rcnn_out"],f"{base}_utilities.gpkg")
            print(f"    {base}...",end=" ",flush=True); dets=[]
            try:
                with rasterio.open(ip) as src:
                    H,W=src.height,src.width; tfm,crs=src.transform,src.crs
                    for y in range(0,H-RP+1,RS):
                        for x in range(0,W-RP+1,RS):
                            p=src.read(window=Window(x,y,RP,RS))
                            if p.shape[0]<3: continue
                            rgb=np.transpose(p[:3],(1,2,0)).astype(np.float32)/255.
                            pt=torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).to(device)
                            with torch.no_grad(): pred_=rcnn([pt[0]])[0]
                            for b,l,s in zip(pred_["boxes"].cpu().numpy(),
                                             pred_["labels"].cpu().numpy(),
                                             pred_["scores"].cpu().numpy()):
                                if s>=0.3 and l in RC:
                                    cx=(b[0]+b[2])/2+x; cy=(b[1]+b[3])/2+y
                                    lon,lat=tfm*(cx,cy)
                                    dets.append({"geometry":Point(lon,lat),"class_name":RC[l],"confidence":float(s)})
                            torch.cuda.empty_cache()
                gdf=(gpd.GeoDataFrame(dets,crs=crs) if dets
                     else gpd.GeoDataFrame(columns=["geometry","class_name","confidence"],crs=crs))
                gdf.to_file(out,driver="GPKG"); print(f"{len(gdf)}")
                summary.append({"Village":base,"Utilities":len(gdf),
                    "Transformers":len(gdf[gdf.class_name=="Transformer"]) if len(gdf) else 0,
                    "Tanks":len(gdf[gdf.class_name=="Tank"]) if len(gdf) else 0,
                    "Wells":len(gdf[gdf.class_name=="Well"]) if len(gdf) else 0})
            except Exception as e:
                print(f"ERROR: {e}")
        pd.DataFrame(summary).to_csv(os.path.join(PATHS["rcnn_out"],"rcnn_summary.csv"),index=False)

    print("\n  STAGE 6 COMPLETE")
    return True

# ============================================================
# STAGE 7 — MERGE / EVALUATE / VISUALISE
# ============================================================

def stage7_merge_evaluate_visualise():
    _banner("STAGE 7 — MERGE / EVALUATE / VISUALISE")
    import numpy as np, pandas as pd
    import matplotlib.pyplot as plt, matplotlib.patches as mpatches
    import geopandas as gpd, torch
    from PIL import Image
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from torch.utils.data import Dataset, DataLoader

    # -- 7a merge
    print("\n[7a] Merging GeoPackages...")
    _mkdirs(PATHS["final_out"])
    ug=sorted(glob.glob(os.path.join(PATHS["unet_out"],"*_features.gpkg")))
    rm={os.path.basename(f).replace("_utilities.gpkg",""):f
        for f in glob.glob(os.path.join(PATHS["rcnn_out"],"*_utilities.gpkg"))}
    vstats=[]; tbc={v:0 for k,v in CLASS_NAMES.items() if k>0}
    for uf in ug:
        v=os.path.basename(uf).replace("_features.gpkg","")
        ud=gpd.read_file(uf)
        if v in rm:
            rd=gpd.read_file(rm[v])
            if ud.crs!=rd.crs: rd=rd.to_crs(ud.crs)
            comb=pd.concat([ud,rd],ignore_index=True)
            print(f"  {v}: {len(ud)}+{len(rd)}->{len(comb)}")
        else:
            comb=ud; print(f"  {v}: unet only ({len(ud)})")
        comb.to_file(os.path.join(PATHS["final_out"],f"{v}_complete.gpkg"),driver="GPKG")
        vc={}
        for cn in tbc:
            cnt=len(comb[comb.class_name==cn]) if "class_name" in comb.columns else 0
            vc[cn]=cnt; tbc[cn]+=cnt
        vstats.append({"Village":v,"Total":len(comb),**vc})
    if vstats:
        pd.DataFrame(vstats).to_csv("final_analysis_summary.csv",index=False)
        print("  Saved: final_analysis_summary.csv")
    tf=sum(tbc.values()); mt={"DuSA U-Net":0,"Faster R-CNN":0}
    for cn,c in tbc.items(): mt[MODEL_SOURCES.get(cn,"DuSA U-Net")]+=c
    print(f"  Total features: {tf:,}")
    for m,c in mt.items(): print(f"    {m}: {c:,} ({c/tf*100 if tf else 0:.1f}%)")

    # -- 7b unet metrics
    print("\n[7b] U-Net validation metrics...")
    class VDS(Dataset):
        def __init__(self):
            self.imgs =sorted(glob.glob(os.path.join(PATHS["patch_images"],"*.png")))
            self.masks=sorted(glob.glob(os.path.join(PATHS["patch_masks"], "*.png")))
            self.tf=A.Compose([A.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),ToTensorV2()])
        def __len__(self): return len(self.imgs)
        def __getitem__(self,i):
            img=np.array(Image.open(self.imgs[i]).convert("RGB"))
            mask=np.clip(np.array(Image.open(self.masks[i])),0,9)
            a=self.tf(image=img,mask=mask); return a["image"],a["mask"].long()

    if os.path.exists(PATHS["unet_model"]) and len(glob.glob(os.path.join(PATHS["patch_images"],"*.png")))>0:
        from models.dusa_unet import DuSA_UNet
        dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ds=VDS(); vs=int(0.2*len(ds))
        _,vds=torch.utils.data.random_split(ds,[len(ds)-vs,vs])
        vl=DataLoader(vds,batch_size=16,shuffle=False,num_workers=4)
        model=DuSA_UNet(n_classes=10).to(dev)
        model.load_state_dict(torch.load(PATHS["unet_model"],map_location=dev))
        model.eval(); cor=tot=0; cc={i:0 for i in range(1,10)}; ct={i:0 for i in range(1,10)}
        with torch.no_grad():
            for imgs,masks in vl:
                imgs,masks=imgs.to(dev),masks.to(dev)
                pred=torch.argmax(model(imgs),1)
                cor+=(pred==masks).sum().item(); tot+=masks.numel()
                for c in range(1,10):
                    mc=(masks==c); pc=(pred==c)
                    cc[c]+=(mc&pc).sum().item(); ct[c]+=mc.sum().item()
        acc=cor/tot*100
        print(f"  Overall: {acc:.2f}%")
        print(f"  {'Class':<20} {'Acc':>9}  {'Pixels':>13}")
        print("  "+"-"*48)
        for c in range(1,10):
            if ct[c]: print(f"  {CLASS_NAMES[c]:<20} {cc[c]/ct[c]*100:>8.2f}%  {ct[c]:>13,}")
            else:     print(f"  {CLASS_NAMES[c]:<20} {'N/A':>9}")
    else:
        print("  Skipping (no model or no patches)")

    # -- 7c charts
    print("\n[7c] Charts...")
    if vstats and tf>0:
        fig,axes=plt.subplots(2,2,figsize=(16,12))
        ax=axes[0,0]; vn=[s["Village"][:18] for s in vstats]; tt=[s["Total"] for s in vstats]
        cols=plt.cm.viridis(np.linspace(0,1,len(vn))); bars=ax.bar(vn,tt,color=cols)
        ax.set_xticklabels(vn,rotation=45,ha="right",fontsize=8)
        ax.set_title("Features per Village"); ax.set_ylabel("Count")
        for b,v in zip(bars,tt): ax.text(b.get_x()+b.get_width()/2,b.get_height()+30,f"{v:,}",ha="center",fontsize=7)
        ax=axes[0,1]; cl=[k for k,v in tbc.items() if v>0]; cc_=[v for v in tbc.values() if v>0]
        if cl: ax.pie(cc_,labels=cl,autopct="%1.1f%%",startangle=90)
        ax.set_title("Feature Distribution")
        ax=axes[1,0]; ax.bar(list(mt.keys()),list(mt.values()),color=["#2ecc71","#e74c3c"])
        ax.set_title("Model Contribution"); ax.set_ylabel("Features")
        for i,(m,c) in enumerate(mt.items()):
            if c: ax.text(i,c+tf*0.005,f"{c:,}",ha="center")
        ax=axes[1,1]; top=sorted([(k,v) for k,v in tbc.items() if v>0],key=lambda x:x[1],reverse=True)[:8]
        if top:
            ns,cs=zip(*top); ax.barh(ns,cs,color="skyblue")
            ax.set_xlabel("Count"); ax.set_title("Top Feature Types")
            for i,(n,c) in enumerate(zip(ns,cs)): ax.text(c+tf*0.002,i,f"{c:,}",va="center",fontsize=8)
        plt.tight_layout(); plt.savefig("final_analysis.png",dpi=150,bbox_inches="tight"); plt.close()
        print("  Saved: final_analysis.png")

    # -- 7d sample patches
    print("\n[7d] Sample patch composite...")
    mfiles=glob.glob(os.path.join(PATHS["patch_masks"],"*.png"))
    if mfiles:
        _mkdirs(PATHS["samples"]); fp={c:[] for c in FEATURE_COLORS}
        for mp in mfiles:
            mask=np.array(Image.open(mp))
            for c in FEATURE_COLORS:
                if np.any(mask==c) and len(fp[c])<4: fp[c].append(mp)
        fig,axes=plt.subplots(3,3,figsize=(15,15))
        fig.suptitle("Feature Class Samples",fontsize=16,fontweight="bold")
        for idx,(cid,info) in enumerate(FEATURE_COLORS.items()):
            ax=axes[idx//3,idx%3]; ps=fp[cid]
            if ps:
                mp=random.choice(ps); ip=mp.replace("masks","images")
                if os.path.exists(ip):
                    img=np.array(Image.open(ip).convert("RGB"))
                    mask=np.array(Image.open(mp)); ov=img.copy(); ov[mask==cid]=info["rgb"]
                    ax.imshow(ov); ax.text(5,25,f"n={len(ps)}",fontsize=8,color="white",backgroundcolor="black")
            ax.set_title(info["name"],fontsize=9,fontweight="bold"); ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(PATHS["samples"],"all_features_composite.png"),dpi=120,bbox_inches="tight")
        plt.close(); print(f"  Saved: {PATHS['samples']}/all_features_composite.png")

    # -- 7e gpkg map
    gpkgs=sorted(glob.glob(os.path.join(PATHS["final_out"],"*.gpkg")))
    if gpkgs:
        print("\n[7e] GeoPackage map...")
        gdf=gpd.read_file(gpkgs[0]); fig,axes=plt.subplots(1,2,figsize=(16,8)); leg=[]
        for info in FEATURE_COLORS.values():
            sub=gdf[gdf.class_name==info["name"]] if "class_name" in gdf.columns else gpd.GeoDataFrame()
            if len(sub): sub.plot(ax=axes[0],color=info["hex"],markersize=2,alpha=0.7)
            leg.append(mpatches.Patch(color=info["hex"],label=info["name"]))
        axes[0].set_title("Extracted Features")
        if leg: axes[0].legend(handles=leg,fontsize=7)
        bldgs=gdf[gdf.class_name.str.contains("Roof",na=False)] if "class_name" in gdf.columns else gdf
        if len(bldgs): bldgs.plot(ax=axes[1],color="red",markersize=2,alpha=0.7)
        axes[1].set_title("Building Footprints"); plt.tight_layout()
        plt.savefig("geopackage_visualization.png",dpi=150,bbox_inches="tight"); plt.close()
        print("  Saved: geopackage_visualization.png")

    print("\n  STAGE 7 COMPLETE")
    return True

# ============================================================
# ENTRY POINT
# ============================================================

STAGE_MAP={
    "1":("Mask generation",      stage1_mask_generation),
    "2":("Patch extraction",     stage2_patch_extraction),
    "3":("Augmentation",         stage3_augmentation),
    "4":("Train U-Net",          stage4_train_unet),
    "5":("Train Faster R-CNN",   stage5_train_rcnn),
    "6":("Inference",            stage6_inference),
    "7":("Merge/Evaluate/Vis",   stage7_merge_evaluate_visualise),
}

def main():
    parser = argparse.ArgumentParser(
        description="SVAMITVA pipeline",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        help="Stage(s): all | 1 | 2 | ... | 7 | 1,2,3"
    )

    args, unknown = parser.parse_known_args()

    stages = list(STAGE_MAP.keys()) if args.stage == "all" else [s.strip() for s in args.stage.split(",")]

    bad = [s for s in stages if s not in STAGE_MAP]
    if bad:
        print(f"Unknown stages: {bad}")
        sys.exit(1)

    print("Running stages:", stages)

    for s in stages:
        name, fn = STAGE_MAP[s]
        fn()

if __name__=="__main__":
    main()    
