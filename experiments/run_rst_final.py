"""
RST Galactic Rotation: Headline Results and Sensitivity Scans

Reproduces all empirical tables in:
"Galactic Rotation from the Compact Fiber: The Gravitational Limit,
R^{16}, and a Zero-Parameter Rotation-Curve Model"

Usage:  cd experiments && python run_rst_final.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from pathlib import Path
import numpy as np, pandas as pd
from sparc_io import list_rotmod_files, load_rotmod_file
from models import rar_exp_benchmark, dft_b_model
from rst_model import rst_rotation_model

KPC, KM = 3.0856775814913673e19, 1000.0
R = 156.444; S_NAT = 4.559e-8; T_NAT = 1.521e-16
A_NAT = S_NAT/T_NAT**2; N_M = 4
A0 = A_NAT*N_M/R**16; L_GL = S_NAT*R**12/KPC

def chi2pp(vo, vm, ev):
    m = np.isfinite(vm)&np.isfinite(vo)&(ev>0)
    return float(np.sum(((vo[m]-vm[m])/ev[m])**2)/m.sum()) if m.sum()>0 else np.nan

def load_all(d):
    gs = []
    for f in sorted(list_rotmod_files(d)):
        g = load_rotmod_file(f)
        if g is None: continue
        if 'Vobs' not in g.df.columns or 'errV' not in g.df.columns: continue
        if len(g.df)<5: continue
        gs.append(g)
    return gs

def rst_param(df, a0, L, p=0.5, d=2.0, eta=0.0, ud=0.5, ub=0.7):
    r_kpc = df['Rad'].values.astype(float)
    vg = np.maximum(df['Vgas'].values.astype(float),0)
    vd = np.maximum(df['Vdisk'].values.astype(float),0)
    vb = np.maximum(df.get('Vbul',pd.Series(np.zeros(len(df)))).values.astype(float),0)
    vbar = np.sqrt(ud*vd**2+ub*vb**2+vg**2)
    r_m = r_kpc*KPC; gbar = (vbar*KM)**2/np.maximum(r_m,1e-30)
    m = np.isfinite(r_kpc)&np.isfinite(gbar)&(gbar>0)
    out = np.full(len(df),np.nan); idx = np.nonzero(m)[0]
    if idx.size==0: return out
    order = np.argsort(r_kpc[idx]); idxs = idx[order]
    r = r_kpc[idx][order]; g = gbar[idx][order]; n = len(r)
    if eta>0 and n>=3:
        lg = np.log(np.maximum(g,1e-30)); c = np.zeros(n)
        for i in range(1,n-1):
            h1=max(r[i]-r[i-1],1e-6); h2=max(r[i+1]-r[i],1e-6)
            c[i]=2*((lg[i+1]-lg[i])/h2-(lg[i]-lg[i-1])/h1)/(h1+h2)
        c[0]=c[1]; c[-1]=c[-2]
        cc=np.clip(c,-5,5); cn=cc/(np.median(np.abs(cc))+1e-30)
        Le = float(L)*np.clip(1+eta*np.tanh(cn),1-eta,1+eta)
    else: Le = np.full(n,float(L))
    rt = np.power(np.maximum(r,0),p); gt = np.zeros(n)
    for i in range(n):
        Li=max(float(Le[i]),1e-6); w=np.exp(-np.abs(rt-rt[i])/Li)
        gt[i]=np.sum(w*g)/np.sum(w)
    ge = np.power(np.maximum(float(a0)*np.maximum(gt,0),0),1.0/d)
    v_ms = np.sqrt(np.maximum((g+ge)*r*KPC,0)); out[idxs]=v_ms/KM
    return out

def main():
    dd = Path(os.path.dirname(__file__))/".."/  "data"/"sparc"/"Rotmod_LTG"
    gals = load_all(dd)
    print(f"Loaded {len(gals)} galaxies | a0={A0:.4e} L={L_GL:.5f} kpc")

    # ── HEADLINE ─────────────────────────────────────────────────────
    print(f"\n{'='*70}\n  HEADLINE: R-DERIVED RST vs MOND\n{'='*70}")
    rows=[]
    for g in gals:
        vo=g.df['Vobs'].values.astype(float); ev=g.df['errV'].values.astype(float)
        r=g.df['Rad'].values.astype(float)
        vr=rst_rotation_model(g.df,A0,L_GL); vm=rar_exp_benchmark(g.df)
        sr=chi2pp(vo,vr,ev); sm=chi2pp(vo,vm,ev)
        no=max(3,len(g.df)//4); si=np.argsort(r); vf=float(np.median(vo[si[-no:]]))
        rows.append(dict(galaxy=g.name.replace('_rotmod.dat',''),npts=len(g.df),
                         Vflat=vf,chi2_RST=sr,chi2_MOND=sm))
    rdf=pd.DataFrame(rows).dropna()
    rst=rdf['chi2_RST'].values; mond=rdf['chi2_MOND'].values
    print(f"RST median={np.median(rst):.2f}, MOND={np.median(mond):.2f}, ratio={np.median(rst)/np.median(mond):.4f}, wins={int(np.sum(rst<mond))}/{len(rdf)}")
    for p in [10,25,50,75,90,95]:
        print(f"  p{p:>2}: RST={np.percentile(rst,p):>8.2f}  MOND={np.percentile(mond,p):>8.2f}")
    print(f"  chi2<2: RST={int(np.sum(rst<2))}, MOND={int(np.sum(mond<2))}")
    print(f"  chi2<5: RST={int(np.sum(rst<5))}, MOND={int(np.sum(mond<5))}")
    rdf['log10_ratio']=np.log10(rst/mond); rdf['winner']=np.where(rst<mond,'RST','MOND')
    csv=os.path.join(os.path.dirname(__file__),'..','rst_sparc_results.csv')
    rdf.to_csv(csv,index=False,float_format='%.4f'); print(f"CSV: {csv}")

    # ── SCANS ────────────────────────────────────────────────────────
    def scan(label, param, values, **kw):
        print(f"\n{'='*70}\n  {label}\n{'='*70}")
        for v in values:
            kw[param]=v; scores=[]
            for g in gals:
                vp=rst_param(g.df,A0,L_GL,**kw)
                s=chi2pp(g.df['Vobs'].values,vp,g.df['errV'].values)
                if np.isfinite(s): scores.append(s)
            med=np.median(scores)
            print(f"  {param}={v}: median={med:.2f} (ratio={med/np.median(mond):.4f})")

    scan("COVERING DEGREE SCAN","d",[1.5,1.8,2.0,2.2,2.5,3.0])
    scan("KERNEL POWER SCAN","p",[0.3,0.4,0.5,0.6,0.7,0.8,1.0])
    scan("CURVATURE MODULATION SCAN","eta",[0.0,0.1,0.2,0.3,0.45,0.55,0.7])
    scan("M/L SENSITIVITY SCAN","ud",[0.3,0.4,0.5,0.6,0.7])

    # ── DFT-B ────────────────────────────────────────────────────────
    print(f"\n{'='*70}\n  DFT-B CALIBRATION\n{'='*70}")
    bg,bm,ba,bl=np.inf,None,None,None
    for ac in np.logspace(np.log10(2e-11),np.log10(8e-10),25):
        for lc in np.linspace(0.3,15.0,25):
            sc=[chi2pp(g.df['Vobs'].values,dft_b_model(g.df,float(ac),float(lc)),g.df['errV'].values) for g in gals]
            sc=[s for s in sc if np.isfinite(s)]
            if sc and np.median(sc)<bg: bg,ba,bl=np.median(sc),float(ac),float(lc)
    print(f"  Best: ac={ba:.3e}, Lc={bl:.2f} kpc, median={bg:.2f}")
    print(f"\nDone.")

if __name__=="__main__":
    main()
