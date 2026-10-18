#!/usr/bin/env python3
"""Controlli minimi ma veri, da eseguire su ogni pull request.

Questo repository non ha test: `pytest` sta fra le dipendenze ma non esiste un
solo file di test. Inventare una suite non e' il compito di questo script;
quello che fa e' verificare tre cose che si rompono per davvero, e che fino a
oggi non erano verificate da niente:

1. ogni modulo del pacchetto si importa. Un import che fallisce e' un difetto,
   e un aggiornamento di dipendenza che rompe un import si vede solo qui
2. config.yaml si carica e produce la struttura che il codice si aspetta
3. l'entry point si importa senza eseguire la pipeline

Non e' un sostituto dei test: non verifica il *comportamento* di niente. Dice
soltanto che il programma si tiene in piedi, che e' esattamente cio' che una
pull request di dipendenze puo' rompere.
"""
from __future__ import annotations

import importlib
import pkgutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

errori: list[str] = []


def controlla(nome: str, azione) -> None:
    try:
        azione()
        print(f"  ok    {nome}")
    except Exception as exc:  # noqa: BLE001 - qui ogni eccezione e' un difetto da riportare
        errori.append(f"{nome}: {type(exc).__name__}: {exc}")
        print(f"  FALLITO {nome}: {type(exc).__name__}: {exc}")


def main() -> int:
    print("1. import di ogni modulo del pacchetto")
    import newsgator

    moduli = [newsgator.__name__] + [
        m.name for m in pkgutil.walk_packages(newsgator.__path__, prefix="newsgator.")
    ]
    for nome in moduli:
        controlla(nome, lambda n=nome: importlib.import_module(n))

    print("\n2. config.yaml")

    # Il file si legge direttamente con yaml, e non e' pignoleria: load_config()
    # *ingoia* un errore di sintassi. Verificato rompendo il file apposta: logga
    # il problema e restituisce i valori per difetto senza alzare nulla, quindi
    # un config.yaml malformato degrada in silenzio e l'intera configurazione
    # dell'operatore viene ignorata. Fino a che resta cosi', il solo modo di
    # accorgersene prima e' parsare il file qui.
    def sintassi() -> None:
        import yaml

        with (ROOT / "config.yaml").open(encoding="utf-8") as fh:
            dati = yaml.safe_load(fh)
        if not isinstance(dati, dict):
            raise TypeError(f"config.yaml non contiene una mappa ma {type(dati).__name__}")
        # Le sezioni lette qui sono quelle *del file*, non quelle del risultato
        # fuso con DEFAULT_CONFIG: controllare il risultato fuso sarebbe un
        # controllo che non puo' fallire, perche' i valori per difetto
        # riempiono sempre ogni chiave.
        attese = {"paths", "rss_feeds", "content_analysis", "feed_processing", "llm", "html"}
        mancanti = attese - set(dati)
        if mancanti:
            raise KeyError(f"sezioni mancanti in config.yaml: {sorted(mancanti)}")

    controlla("sintassi e sezioni di config.yaml", sintassi)

    def caricatore() -> None:
        from newsgator.config import load_config

        cfg = load_config()
        if not isinstance(cfg, dict):
            raise TypeError(f"load_config() ha restituito {type(cfg).__name__}, atteso dict")

    controlla("load_config() gira", caricatore)

    print("\n3. import dell'entry point")
    controlla("main.py", lambda: importlib.import_module("newsgator").main)

    print(f"\n  moduli controllati: {len(moduli)}   errori: {len(errori)}")
    if errori:
        print("\n  RIEPILOGO DEGLI ERRORI:")
        for e in errori:
            print(f"    - {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
