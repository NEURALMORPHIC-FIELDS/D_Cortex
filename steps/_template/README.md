# Step NN — v15.X — `descriere_scurta`

**Status**: [ÎN LUCRU / SIGILAT]. Verdict: [PASS / PARTIAL / FAIL / —]

## Problema atacată

Descrie în 2-4 propoziții ce cauză concretă vrei să închizi. Leagă de output-ul step-ului predecesor (ce metrică rămânea neatinsă, ce diagnostic a arătat cauza).

## Ipoteza arhitecturală

Ce schimbare structurală ar trebui să rezolve problema. O propoziție. Dacă nu poți s-o spui într-o propoziție, nu e ipoteză, e colecție de idei.

## Schimbări față de step predecesor

Listă explicită, granulară:

1. **Componentă nouă**: `X` — ce face
2. **Modificare**: `Y` extinde `Z` (nu modifică, extinde)
3. **Eliminare**: dacă ceva dispare, spui explicit
4. ...

### Invarianți menținuți

- [componentă]: neatinsă
- [componentă]: neatinsă
- ...

## Gates de acceptare

Praguri numerice, nu calitative. Exemplu:

- Gate 0: trusted regression byte-identical (before/after)
- Gate 1: `wrong_commit_rate` ≤ 0.02 pe toate 5 familii
- Gate N: [metric specific step-ului] [operator] [valoare]

## Rezultate

Tabel cu delta vs predecesor, nu valori absolute izolate.

| Metric | Predecesor | Acest step | Delta |
|---|---:|---:|---:|
| ... | ... | ... | ... |

## Scope guard (explicit ce NU atingem)

Listă strictă. Dacă un pas viitor are regresie în aceste zone, vina e aici.

- ...

## Mediu de execuție

Unde rulează. Parametri relevanți (seed-uri, n_trials, hyperparameters).

## Artefacte

Unde apar rezultatele (path-uri în Drive / local).

## Succesor prevăzut

Ce urmează, dacă acest pas trece. Dacă pică, ce plan B.
