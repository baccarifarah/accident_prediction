"""Prédiction du risque d'accident par réseau bayésien (pgmpy).

Entrées : historiques de positions des véhicules suivis, en mètres dans la vue
de dessus (sortie de ViewTransformer), sous forme de (frame, x, y).
Convention : y = axe de la route (longitudinal), x = axe transversal.

Pipeline par paire de véhicules proches :
    cinématique -> distance, headway, TTC, vitesse, trajectoire
    -> discrétisation (THRESHOLDS) -> réseau bayésien -> P(accident)
    -> niveau de risque (faible / moyen / élevé)

Les tables de probabilités (CPT) sont définies par expertise : aucun jeu de
données d'accidents étiqueté n'est disponible. La structure permet de les
apprendre plus tard avec `model.fit(dataframe)`.
"""

from functools import lru_cache
from itertools import combinations

import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

# Seuils de discrétisation et de classification, à ajuster selon la scène.
THRESHOLDS = {
    "ttc_s": (1.5, 3.0),  # critique < 1.5 s <= alerte < 3 s <= sur
    "headway_s": (1.0, 2.0),  # critique < 1 s <= courte < 2 s <= sure (règle des 2 s)
    "speed_kmh": (50.0, 110.0),  # faible < 50 <= normale < 110 <= elevee
    "accel_ms2": 3.0,  # trajectoire brusque (freinage / écart) au-delà
    "risk_p": (0.30, 0.60),  # faible < 0.30 <= moyen < 0.60 <= élevé
}

VEHICLE_LENGTH_M = 4.5  # zone de collision longitudinale
VEHICLE_WIDTH_M = 2.0  # zone de collision transversale
LANE_WIDTH_M = 3.5  # au-delà : pas dans la même voie, pas de headway
PAIR_MAX_DISTANCE_M = 60.0  # paires plus éloignées ignorées

RISK_LEVELS = ("faible", "moyen", "élevé")

STATES = {
    "TTC": ["critique", "alerte", "sur"],
    "Distance": ["critique", "courte", "sure"],
    "Vitesse": ["faible", "normale", "elevee"],
    "Trajectoire": ["stable", "brusque"],
    "Conflit": ["non", "oui"],
    "Accident": ["non", "oui"],
}


def _cpd(variable, p_yes, evidence=None):
    """CPT binaire (non/oui) à partir de P(oui) pour chaque combinaison des parents."""
    p_yes = np.asarray(p_yes, dtype=float)
    return TabularCPD(
        variable,
        2,
        [1 - p_yes, p_yes],
        evidence=evidence,
        evidence_card=[len(STATES[e]) for e in evidence] if evidence else None,
        state_names={v: STATES[v] for v in [variable, *(evidence or [])]},
    )


def _prior(variable, probabilities):
    return TabularCPD(
        variable,
        len(STATES[variable]),
        [[p] for p in probabilities],
        state_names={variable: STATES[variable]},
    )


def build_network():
    """Réseau : TTC, Distance -> Conflit ; Conflit, Vitesse, Trajectoire -> Accident."""
    model = DiscreteBayesianNetwork(
        [
            ("TTC", "Conflit"),
            ("Distance", "Conflit"),
            ("Conflit", "Accident"),
            ("Vitesse", "Accident"),
            ("Trajectoire", "Accident"),
        ]
    )
    # A priori des racines (trafic autoroutier courant). Toutes les racines sont
    # observées à l'inférence, ces valeurs servent à la validité du modèle.
    priors = [
        _prior("TTC", [0.05, 0.15, 0.80]),
        _prior("Distance", [0.10, 0.30, 0.60]),
        _prior("Vitesse", [0.20, 0.60, 0.20]),
        _prior("Trajectoire", [0.85, 0.15]),
    ]
    # P(Conflit=oui | TTC, Distance). Colonnes : TTC (lent) x Distance (rapide).
    #        Distance:  critique courte sure
    conflit = _cpd(
        "Conflit",
        [
            0.97, 0.95, 0.90,  # TTC critique
            0.75, 0.60, 0.45,  # TTC alerte
            0.40, 0.15, 0.02,  # TTC sur
        ],
        evidence=["TTC", "Distance"],
    )  # fmt: skip
    # P(Accident=oui | Conflit, Vitesse, Trajectoire).
    #        Trajectoire: stable brusque
    accident = _cpd(
        "Accident",
        [
            0.005, 0.03,  # Conflit non, Vitesse faible
            0.01, 0.06,  # Conflit non, Vitesse normale
            0.03, 0.12,  # Conflit non, Vitesse elevee
            0.40, 0.60,  # Conflit oui, Vitesse faible
            0.65, 0.85,  # Conflit oui, Vitesse normale
            0.80, 0.95,  # Conflit oui, Vitesse elevee
        ],
        evidence=["Conflit", "Vitesse", "Trajectoire"],
    )  # fmt: skip
    model.add_cpds(*priors, conflit, accident)
    assert model.check_model()
    return model


@lru_cache(maxsize=1)
def _inference():
    return VariableElimination(build_network())


# ponytail: 3*3*3*2 = 54 combinaisons possibles, pgmpy n'est appelé qu'une fois par combinaison
@lru_cache(maxsize=None)
def accident_probability(ttc, distance, vitesse, trajectoire):
    """P(Accident=oui) étant donné les états discrets observés."""
    result = _inference().query(
        ["Accident"],
        evidence={
            "TTC": ttc,
            "Distance": distance,
            "Vitesse": vitesse,
            "Trajectoire": trajectoire,
        },
        show_progress=False,
    )
    return float(result.get_value(Accident="oui"))


def _bucket(value, bounds, states):
    """Premier état dont la borne dépasse value, sinon le dernier."""
    for bound, state in zip(bounds, states):
        if value < bound:
            return state
    return states[-1]


def discretize(ttc, headway, speed_kmh, accel):
    return (
        _bucket(ttc, THRESHOLDS["ttc_s"], STATES["TTC"]),
        _bucket(headway, THRESHOLDS["headway_s"], STATES["Distance"]),
        _bucket(speed_kmh, THRESHOLDS["speed_kmh"], STATES["Vitesse"]),
        "brusque" if accel >= THRESHOLDS["accel_ms2"] else "stable",
    )


def risk_level(probability):
    """Indice dans RISK_LEVELS : 0 faible, 1 moyen, 2 élevé."""
    low, high = THRESHOLDS["risk_p"]
    return 0 if probability < low else 1 if probability < high else 2


def kinematics(history, fps):
    """Position, vitesse (m/s, vecteur) et accélération (m/s²) depuis (frame, x, y).

    Retourne None si l'historique couvre moins d'une demi-seconde.
    """
    h = np.asarray(history, dtype=float)
    if len(h) < 3 or (h[-1, 0] - h[0, 0]) < fps / 2:
        return None
    t, p = h[:, 0] / fps, h[:, 1:]
    mid = len(h) // 2
    v = (p[-1] - p[0]) / (t[-1] - t[0])
    v1 = (p[mid] - p[0]) / (t[mid] - t[0])
    v2 = (p[-1] - p[mid]) / (t[-1] - t[mid])
    accel = np.linalg.norm(v2 - v1) / ((t[-1] - t[0]) / 2)
    return {"position": p[-1], "velocity": v, "speed_kmh": np.linalg.norm(v) * 3.6, "accel": accel}


def time_to_collision(rel_position, rel_velocity):
    """Temps avant que la paire entre en zone de collision, à vitesse constante.

    Zone de collision = rectangle longueur x largeur, approximé par une ellipse :
    on dilate l'axe x de L/W puis on résout |p + v t| = L. inf si aucune collision.
    """
    scale = np.array([VEHICLE_LENGTH_M / VEHICLE_WIDTH_M, 1.0])
    p, v = rel_position * scale, rel_velocity * scale
    a, b, c = v @ v, 2 * p @ v, p @ p - VEHICLE_LENGTH_M**2
    if c <= 0:
        return 0.0  # déjà en contact
    disc = b * b - 4 * a * c
    if a == 0 or disc < 0:
        return np.inf
    t = (-b - np.sqrt(disc)) / (2 * a)
    return t if t > 0 else np.inf  # c > 0 : les deux racines ont le même signe


def assess_pair(ki, kj):
    """Caractéristiques, états discrets et P(accident) pour une paire, ou None si trop loin."""
    rel_p = kj["position"] - ki["position"]
    distance = float(np.linalg.norm(rel_p))
    if distance > PAIR_MAX_DISTANCE_M:
        return None
    ttc = time_to_collision(rel_p, kj["velocity"] - ki["velocity"])
    # Headway seulement dans la même voie : écart longitudinal / vitesse du suiveur.
    if abs(rel_p[0]) < LANE_WIDTH_M:
        follower = ki if np.dot(rel_p, ki["velocity"]) > 0 else kj
        gap = max(abs(rel_p[1]) - VEHICLE_LENGTH_M, 0.0)
        headway = gap / max(follower["speed_kmh"] / 3.6, 0.1)
    else:
        headway = np.inf
    speed = max(ki["speed_kmh"], kj["speed_kmh"])
    accel = max(ki["accel"], kj["accel"])
    states = discretize(ttc, headway, speed, accel)
    probability = accident_probability(*states)
    return {
        "distance": distance,
        "ttc": ttc,
        "headway": headway,
        "states": states,
        "probability": probability,
        "level": risk_level(probability),
    }


def assess_frame(tracks, fps):
    """Évalue toutes les paires de véhicules visibles.

    tracks : {tracker_id: historique (frame, x, y)}.
    Retourne (cinematique, risque_vehicule, paires) :
      cinematique     {id: dict de kinematics()}
      risque_vehicule {id: (niveau, probabilité)} = pire paire du véhicule
      paires          liste de dicts assess_pair() + "ids", triée par probabilité décroissante
    """
    kin = {tid: k for tid, h in tracks.items() if (k := kinematics(h, fps)) is not None}
    vehicle_risk = {tid: (0, 0.0) for tid in kin}
    pairs = []
    for i, j in combinations(kin, 2):
        pair = assess_pair(kin[i], kin[j])
        if pair is None:
            continue
        pair["ids"] = (i, j)
        pairs.append(pair)
        for tid in (i, j):
            if pair["probability"] > vehicle_risk[tid][1]:
                vehicle_risk[tid] = (pair["level"], pair["probability"])
    pairs.sort(key=lambda p: p["probability"], reverse=True)
    return kin, vehicle_risk, pairs


def _track(x, y, vx, vy, fps=30, seconds=1.0, ax=0.0, ay=0.0):
    """Historique synthétique (frame, x, y) pour les vérifications."""
    t = np.arange(int(fps * seconds)) / fps
    return [(f, x + vx * s + ax * s * s / 2, y + vy * s + ay * s * s / 2) for f, s in enumerate(t)]


if __name__ == "__main__":
    model = build_network()
    for cpd in model.get_cpds():
        if cpd.variable in ("Conflit", "Accident"):
            print(cpd)

    # Vitesses en m/s : 25 m/s = 90 km/h.
    # 1. Collision frontale imminente, même voie : élevé.
    _, risk, pairs = assess_frame({1: _track(0, 0, 0, 25), 2: _track(0, 70, 0, -25)}, 30)
    print("frontale :", pairs[0])
    assert pairs[0]["level"] == 2

    # 2. Voies parallèles, même vitesse, côte à côte : faible (pas de TTC).
    _, risk, pairs = assess_frame({1: _track(0, 0, 0, 25), 2: _track(3.5, 5, 0, 25)}, 30)
    print("parallèle :", pairs[0])
    assert pairs[0]["ttc"] == np.inf and pairs[0]["level"] == 0

    # 3. Suivi à ~0.7 s, le véhicule de devant freine fort : au moins moyen.
    _, risk, pairs = assess_frame({1: _track(0, 0, 0, 25), 2: _track(0, 22, 0, 25, ay=-6)}, 30)
    print("freinage :", pairs[0])
    assert pairs[0]["level"] >= 1

    # 4. Véhicules éloignés : paire ignorée.
    _, risk, pairs = assess_frame({1: _track(0, 0, 0, 25), 2: _track(0, 200, 0, 25)}, 30)
    assert pairs == [] and risk == {1: (0, 0.0), 2: (0, 0.0)}

    print("OK")
