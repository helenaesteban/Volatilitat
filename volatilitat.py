import numpy as np

def volatilitat(
    preus=None,
    *,
    returns=None,
    method="log",          # "log" (recomanat) o "simple"
    ddof=1,                # 1 = estimador mostral; 0 = poblacional
    annualize=True,        # anualitzar o no
    periods_per_year=252,  # 252 (diari), 52 (setmanal), 12 (mensual), etc.
    check_positive=True,   # valida que els preus siguin >0 si method="log"/"simple"
    dropna=True            # ignora NaN/inf si n'hi ha
):
    """
    Calcula la volatilitat (desviació estàndard) d'una sèrie de rendiments o preus.

    Paràmetres:
    - preus: seqüència de preus (llista/ndarray/pandas Series). Opcional si passes 'returns'.
    - returns: seqüència de rendiments ja calculats (opc.) — s'ignora 'preus' si es dona.
    - method: "log" (np.diff(np.log(preus))) o "simple" ((P_t-P_{t-1})/P_{t-1}).
    - ddof: 1 per mostral (estàndard en finances), 0 per poblacional.
    - annualize: si True, multiplica per sqrt(periods_per_year).
    - periods_per_year: 252, 52, 12, etc., segons la freqüència dels rendiments.
    - check_positive: valida preus > 0 per evitar problemes amb log o divisions.
    - dropna: elimina NaN/inf dels rendiments abans del càlcul.

    Retorna:
    - volatilitat (float): en unitats del mateix període que els rendiments;
      si annualize=True, anualitzada.
    """
    # Entrades
    if returns is None:
        if preus is None:
            raise ValueError("Cal passar 'preus' o bé 'returns'.")
        x = np.asarray(preus, dtype=float)

        if check_positive and np.any(x <= 0):
            raise ValueError("Hi ha preus ≤ 0; no és vàlid per a rendiments log o simples.")

        if x.size < 2:
            raise ValueError("Es necessiten com a mínim 2 preus per calcular rendiments.")

        if method == "log":
            rets = np.diff(np.log(x))
        elif method == "simple":
            rets = np.diff(x) / x[:-1]
        else:
            raise ValueError("method ha de ser 'log' o 'simple'.")
    else:
        rets = np.asarray(returns, dtype=float)

    # Neteja
    if dropna:
        rets = rets[np.isfinite(rets)]

    if rets.size == 0:
        raise ValueError("Cap rendiment vàlid després de netejar NaN/inf.")

    # Desviació estàndard
    vol = np.std(rets, ddof=ddof)

    # Anualització
    if annualize:
        vol *= np.sqrt(periods_per_year)

    return vol 


if __name__ == "__main__":
    preus = [100, 102, 101, 105, 107, 106]
    vol_diaria = volatilitat(preus, method="log", ddof=1, annualize=False)
    vol_anual  = volatilitat(preus, method="log", ddof=1, annualize=True, periods_per_year=252)
    print(f"Volatilitat diària (log): {vol_diaria:.6f}")
    print(f"Volatilitat anualitzada (log): {vol_anual:.6f}")