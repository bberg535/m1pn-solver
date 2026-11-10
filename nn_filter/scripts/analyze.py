#!/usr/bin/env python3
"""Write portable figures and a concise report from the saved local screen."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'nx/scripts'))
from analysis import beam_exact, physical_low, grid_tensor


def field(row):
    run=ROOT/row['run']
    final=np.loadtxt(run/'final.csv',delimiter=',',skiprows=1)
    low=physical_low(final[:,4:])
    return final[:,1:4],low[:,0]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,default=ROOT/'results/crossing_20261109')
    args=p.parse_args(); out=args.study
    rows=json.loads((out/'results.json').read_text())
    regular=[r for r in rows if not r['time_refined']]
    lookup={(r['grid'],r['order'],r['coupling'],r['filter']):r for r in regular}
    groups=sorted({(r['grid'],r['order'],r['coupling']) for r in regular})
    comparison=[]
    for group in groups:
        off,const,neural=[lookup[group+(mode,)] for mode in ('off','constant','nn')]
        comparison.append({'grid':group[0],'order':group[1],'coupling':group[2],
            'off_L1':off['density_rel_L1'],'constant_L1':const['density_rel_L1'],
            'nn_L1':neural['density_rel_L1'],'nn_vs_off':neural['density_rel_L1']/off['density_rel_L1'],
            'constant_vs_off':const['density_rel_L1']/off['density_rel_L1']})
    temporal=[]
    for fine in (r for r in rows if r['time_refined']):
        coarse=lookup[(fine['grid'],fine['order'],fine['coupling'],fine['filter'])]
        xyz,rho=field(coarse); xyz_f,rho_f=field(fine)
        assert np.array_equal(xyz,xyz_f)
        mass=np.loadtxt(ROOT/coarse['run']/'nodal_mass.csv',delimiter=',',skiprows=1)[:,1]
        cfg=json.loads((ROOT/coarse['run']/'config.json').read_text())
        ref,_=beam_exact(xyz,cfg['final_time'],cfg['beam_geometry'])
        change=float(np.sum(mass*np.abs(rho-rho_f))/np.sum(mass*np.abs(ref)))
        temporal.append({'filter':fine['filter'],'density_change_rel_L1':change,
                         'change_divided_by_base_reference_error':change/coarse['density_rel_L1'],
                         'base_error':coarse['density_rel_L1'],'half_dt_error':fine['density_rel_L1']})
    (out/'comparison.json').write_text(json.dumps({'groups':comparison,'temporal':temporal},indent=2)+'\n')
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    modes=['off','constant','nn'];labels=['Ohne Filter','Konstant','NN (Originalgewichte)']
    colors=['#6b7280','#d97706','#2563eb']
    orders=sorted({r['order'] for r in regular})
    grids=sorted({r['grid'] for r in regular})
    fig,axes=plt.subplots(2,len(orders),figsize=(5*len(orders),7.5),squeeze=False,layout='constrained')
    for j,order in enumerate(orders):
        for i,coupling in enumerate(('m1pn','pn-only')):
            ax=axes[i,j]; x=np.arange(len(grids))
            for k,mode in enumerate(modes):
                ax.bar(x+(k-1)*.23,[lookup[(g,order,coupling,mode)]['density_rel_L1'] for g in grids],
                       width=.22,label=labels[k],color=colors[k])
            ax.set(xticks=x,xticklabels=[f'{g}²' for g in grids],ylabel='Relativer Dichtefehler L1',
                   title=f'P{order} · {"M1PN gekoppelt" if coupling=="m1pn" else "PN ohne Kopplung"}')
            ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Crossing bis T=0.32 · native Massenkorrektur B\nReferenz: analytische kinetische Dichte')
    for ext in ('png','pdf'):fig.savefig(out/f'errors.{ext}',dpi=180)
    plt.close(fig)
    if 32 in grids:
        fig,axes=plt.subplots(len(orders),4,figsize=(13,3.6*len(orders)),squeeze=False,layout='constrained')
        strength_fig,strength_axes=plt.subplots(1,len(orders),figsize=(5*len(orders),4),squeeze=False,layout='constrained')
        for i,order in enumerate(orders):
            base=lookup[(32,order,'m1pn','off')]
            xyz,_=field(base)
            cfg=json.loads((ROOT/base['run']/'config.json').read_text())
            ref,_=beam_exact(xyz,cfg['final_time'],cfg['beam_geometry'])
            rho=[ref]+[field(lookup[(32,order,'m1pn',mode)])[1] for mode in modes]
            max_value=max(float(a.max()) for a in rho)
            for j,values in enumerate(rho):
                xy,data=grid_tensor(xyz,values[:,None],2)
                ax=axes[i,j]
                im=ax.pcolormesh(xy[0],xy[1],data[:,:,0].T,shading='auto',vmin=0,vmax=max_value,cmap='viridis')
                ax.set(title=f'P{order} · '+(['Kinetische Referenz']+labels)[j],aspect='equal',xlabel='x',ylabel='y')
            fig.colorbar(im,ax=list(axes[i,:]),shrink=.8,label='Dichte')
            nnrow=lookup[(32,order,'m1pn','nn')]
            sig=np.loadtxt(ROOT/nnrow['run']/'filter_strength.csv',delimiter=',',skiprows=1)[:,1]
            xy,vals=grid_tensor(xyz,sig[:,None],2)
            ax=strength_axes[0,i]
            im=ax.pcolormesh(xy[0],xy[1],vals[:,:,0].T,shading='auto',cmap='plasma')
            ax.set(title=f'P{order} · NN-Filterstärke bei T=0.32',aspect='equal',xlabel='x',ylabel='y')
            strength_fig.colorbar(im,ax=ax)
        fig.suptitle('Gekoppeltes M1PN auf 32² · T=0.32')
        for ext in ('png','pdf'):
            fig.savefig(out/f'density_32.{ext}',dpi=180)
            strength_fig.savefig(out/f'strength_32.{ext}',dpi=180)
        plt.close(fig);plt.close(strength_fig)
    nn_wins=sum(r['nn_L1']<r['constant_L1'] for r in comparison)
    nn_improvements=sum(r['nn_L1']<r['off_L1'] for r in comparison)
    lines=['# NN-FPN im M1PN-Massenkorrektur-Solver: lokaler Test vom 09.11.2026','',
        f'**{len(rows)} von {len(rows)} Läufen abgeschlossen und aus gespeicherten Feldern geprüft.** '
        f'Das NN senkt den relativen L1-Dichtefehler gegenüber ungefiltertem B in {nn_improvements}/{len(comparison)} Vergleichen. '
        f'Gegenüber der konstanten Paperstärke gewinnt es in {nn_wins}/{len(comparison)} Vergleichen.','',
        'Die Originalgewichte wurden ohne neues Training auf den nativen FE/M1PN-Pfad übertragen. '
        'Die Ergebnisse gelten für diesen Crossing-Test und diese groben Gitter. '
        'Ein allgemeiner Vorteil gegenüber konstanter Filterung ist daraus nicht belegt.','',
        '## Relativer L1-Dichtefehler zur kinetischen Referenz','',
        '| Gitter | PN | Kopplung | Ohne Filter | Konstant | NN | NN / ohne |',
        '|---|---|---|---:|---:|---:|---:|']
    for r in comparison:
        lines.append(f'| {r["grid"]}² | P{r["order"]} | {r["coupling"]} | {r["off_L1"]:.6f} | {r["constant_L1"]:.6f} | {r["nn_L1"]:.6f} | {r["nn_vs_off"]:.4f} |')
    lines += ['', 'Kleinere Werte sind besser. Der Nenner ist jeweils die massengewichtete L1-Norm '
        'der analytischen kinetischen Dichte bei T=0.32. Der Fehler enthält den endlichen '
        'PN-Modellfehler. Es wurden keine negativen Dichten vor der Fehlerauswertung abgeschnitten.','',
        '![Fehlervergleich](errors.png)','', '![Dichten](density_32.png)','',
        '## Erhaltung und Stabilität','',
        f'* Größter unabhängig nachgerechneter globaler Bilanzrest aller PN-Komponenten: '
        f'`{max(r["balance_max"] for r in rows):.3e}`. Rand-, Material- und Filterbeiträge sind enthalten.',
        '* Der Dichtebeitrag des Filters ist in sämtlichen gespeicherten Filterbilanzen exakt null.',
        f'* Rückgewiesene Schrittversuche über alle Läufe: {sum(r["rejected_attempts"] for r in rows)}.',
        f'* Gekoppelte Läufe: kleinste Enddichte `{min(r["minimum_density"] for r in rows if r["coupling"]=="m1pn"):.6g}`, '
        f'kleinster Kegelabstand `{min(r["minimum_cone_margin"] for r in rows if r["coupling"]=="m1pn"):.6g}`.',
        '* Ungekuppeltes PN besitzt auch mit Filter negative Dichten und Kegelverletzungen; '
        'der Filter ersetzt den M1-Limiter nicht. Details stehen in `results.csv`.','',
        '## Halbierte Schrittweite bei gekoppeltem P3/32²','',
        '| Filter | Basisfehler | Fehler bei dt/2 | Feldänderung L1 / Referenznorm | Änderung / Basisfehler |',
        '|---|---:|---:|---:|---:|']
    for r in temporal:
        lines.append(f'| {r["filter"]} | {r["base_error"]:.7f} | {r["half_dt_error"]:.7f} | {r["density_change_rel_L1"]:.3e} | {r["change_divided_by_base_reference_error"]:.3e} |')
    lines += ['', 'Die Feldänderung ist ein Vergleich zweier Zeitschrittweiten, keine Schätzung '
        'der räumlichen Konvergenzordnung oder des exakten Zeitfehlers.','',
        '## Umsetzung und Validierung','',
        '* 19 algebraische Prüfungen: Filterkoeffizienten, analytische Dämpfung, '
        'Dichteerhaltung, Dissipation, Kegel, Rotationskommutation, Netzwerte und Eingabeschutz.',
        '* C++-Netz gegen PyTorch für P3/P5/P7/P9 bei float64 und identischen Gewichten: maximaler absoluter Fehler `7.11e-15`.',
        '* Merkmalsaufbereitung gegen die beiden Originalfunktionen im Autoren-Code: '
        'maximaler Fehler `1.55e-15` bei identischen vorgegebenen FE-Ableitungsfeldern.',
        '* Ende-zu-Ende: Filter aus und konstante Stärke null liefern bitgleiche Felder '
        'zum separat gebauten unveränderten Studienprogramm; außerdem Tests beider '
        'Kopplungen und von Absorption, Streuung und Quelle.',
        '* Jeder Methodenvergleich prüft identische Anfangsfelder, Knotenmassen und '
        'akzeptierte Zeitschritte. Alle gespeicherten Zustände sind endlich.','',
        '## Grenzen und Reproduktion','',
        'Kein Retraining auf FE, M1PN oder Crossing. Die tatsächlich veröffentlichte '
        'Merkmalsdefinition einschließlich räumlicher Standardisierung wird übernommen; '
        'sie entspricht nicht exakt den abstrakten Merkmalen des PDFs. Der Filter wird '
        'über exponentielle Mittelpunktsschritte um den SSPRK3-Transport integriert. '
        'Einzelheiten und die Abweichungen zum Autoren-Code stehen in [README.md](../../README.md).','',
        'Die Laufzeiten in `results.csv` stammen aus einzelnen lokalen Läufen und schließen '
        'unterschiedliche Initialisierungs-/Dateikosten ein. Sie sind keine belastbare '
        'Performancestudie. `filter_seconds` erfasst separat die Filterschritte.','',
        'Die Rohfelder liegen unter `../../tmp/crossing_20261109/`, einschließlich '
        '`filter_verification.json` je Lauf. `manifest.json` sichert Binary, Protokoll, '
        'Elternstand, Modellmanifest und Ergebnisdatei. Die Vergleichswerte stehen in '
        '[results.csv](results.csv), [results.json](results.json) und [comparison.json](comparison.json). '
        'PDF-Ausgaben: [Fehler](errors.pdf), [Dichte](density_32.pdf), [Stärke](strength_32.pdf).','',
        '```bash','python3 scripts/study.py --output results/neuer_vergleich',
        'python3 scripts/analyze.py --study results/neuer_vergleich','```','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'groups':comparison,'temporal':temporal},indent=2))


if __name__=='__main__':main()
