BITS = [
      "RESET_DENS", "RESET_TAU", "RESET_STILDE", "RESET_ENTROPY", "RESET_YE",
      "SIG_RHO_TOO_LOW", "SIG_RHO_TOO_HIGH", "SIG_EPS_TOO_LOW", "SIG_EPS_TOO_HIGH",
      "SIG_YE_TOO_LOW", "SIG_YE_TOO_HIGH", "SIG_ENT_TOO_LOW", "SIG_ENT_TOO_HIGH",
      "SIG_TEMP_TOO_LOW", "SIG_TEMP_TOO_HIGH", "SIG_PRESS_TOO_LOW", "SIG_PRESS_TOO_HIGH",
      "SIG_VEL_TOO_HIGH", "SIG_SIGMA_TOO_HIGH",
      "ENT_BACKUP_USED", "ATMO_RESET", "T_FLOORED",
  ]

def decode_c2p_err(v):
    v = int(v)
    return [BITS[i] for i in range(len(BITS)) if v & (1 << i)]

def print_c2p_err(v):
    flags = decode_c2p_err(v)
    print(f"c2p_err = {int(v)}  (0b{int(v):022b})")
    if not flags:
        print("  (clean)")
    else:
        for f in flags:
            print(f"  {f}")