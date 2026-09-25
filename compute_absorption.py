# a one time script to calculate db/m attenuation values for each frequency due to air absorption

import numpy as np

# ISO 9613-1 reference constants (fixed, don't change these)
T0  = 293.15    # K, reference temperature (20 °C)
T01 = 273.16    # K, triple-point isotherm
PR  = 101.325   # kPa, reference pressure

def air_absorption_db_per_m(f, T_c=20.0, rh=50.0, p_kpa=101.325):

    T = T_c + 273.15
    pr = p_kpa / PR                      # pa / pr

    # water vapour molar concentration (%)
    C = -6.8346 * (T01 / T) ** 1.261 + 4.6151
    h = rh * 10 ** C / pr

    # relaxation frequencies (Hz)
    frO = pr * (24 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    frN = pr * (T / T0) ** -0.5 * (
        9 + 280 * h * np.exp(-4.170 * ((T / T0) ** (-1 / 3) - 1))
    )

    f = np.asarray(f, dtype=float)
    return 8.686 * f**2 * (
        1.84e-11 / pr * np.sqrt(T / T0)
        + (T / T0) ** -2.5 * (
            0.01275 * np.exp(-2239.1 / T) / (frO + f**2 / frO)
            + 0.1068  * np.exp(-3352.0 / T) / (frN + f**2 / frN)
        )
    )

f = [125, 250, 500, 1000, 1400, 2000, 2800, 4000, 5600, 8000, 11300, 16000]
a = {float(freq): round(float(air_absorption_db_per_m(freq)), 4) for freq in f}
print(a)