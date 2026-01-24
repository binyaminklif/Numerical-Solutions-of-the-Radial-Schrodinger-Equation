import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from scipy.integrate import cumulative_trapezoid
harmonic_folder_path = r"C:\Users\bibis\OneDrive\שולחן העבודה\פיזיקה\שנה ג-bibish\קוונטים 2\numerical_exercise\Harmonic_result"
electric_folder_path = r"C:\Users\bibis\OneDrive\שולחן העבודה\פיזיקה\שנה ג-bibish\קוונטים 2\numerical_exercise\Electric_result"
def solve_finite_difference(r_grid, V_grid, l):
    K = len(r_grid)
    delta = r_grid[1] - r_grid[0]

    # Effective potential: W(r) = V(r) + l(l+1)/(2*r^2)
    W = V_grid + (l * (l + 1)) / (2 * r_grid ** 2)

    # Initialize Matrix M
    M = np.zeros((K, K))

    # Fill M based on the finite difference equations from the text
    main_diag = (1 / delta ** 2) + W
    off_diag = -1 / (2 * delta ** 2)

    for k in range(K):
        M[k, k] = main_diag[k]
        if k > 0:
            M[k, k - 1] = off_diag
        if k < K - 1:
            M[k, k + 1] = off_diag

    # Solve for eigenvalues (energies) and eigenvectors (wavefunctions)
    energies, wavefunctions = eigh(M)

    # Calculate the norms
    norms = np.zeros(wavefunctions.shape[1])
    for k in range(1, K):
        norms += 0.5 * delta * (np.abs(wavefunctions[k - 1, :]) ** 2 + np.abs(wavefunctions[k, :]) ** 2)

    # 2. Normalize U by dividing each column by the square root of its norm
    wavefunctions_normalized = wavefunctions / np.sqrt(norms)
    # Return sorted real parts as requested
    return energies, wavefunctions_normalized  #






def solve_numerov(r_grid, V_grid, l):
    K = len(r_grid)
    delta = r_grid[1] - r_grid[0]

    # Effective potential: W(r) = V(r) + l(l+1)/(2*r^2)
    W = V_grid + (l * (l + 1)) / (2 * r_grid ** 2)

    # Initialize matrices H and N
    H = np.zeros((K, K))
    N = np.zeros((K, K))

    # Constants for matrix elements based on the Numerov derivation
    inv_2delta2 = 1 / (2 * (delta ** 2))
    one_twelfth = 1 / 12
    ten_twelfths = 10 / 12

    # Fill H and N matrices using equations from the text
    for k in range(K):
        # Main diagonal
        H[k, k] = (1 / ( delta ** 2)) + (ten_twelfths * W[k])  #
        N[k, k] = ten_twelfths  #

        # Off diagonals
        if k > 0:  # k-1
            H[k, k - 1] = -inv_2delta2 + (one_twelfth * W[k - 1])  #
            N[k, k - 1] = one_twelfth  #
        if k < K - 1:  # k+1
            H[k, k + 1] = -inv_2delta2 + (one_twelfth * W[k + 1])  #
            N[k, k + 1] = one_twelfth  #

    # Solve the generalized eigenvalue problem: H*u = epsilon * N*u
    # eigh with the 'b' parameter solves generalized problems
    energies, wavefunctions = eigh(H, N)  #

    # Calculate the norms
    norms = np.zeros(wavefunctions.shape[1])
    for k in range(1, K):
        norms += 0.5 * delta * (np.abs(wavefunctions[k - 1, :]) ** 2 + np.abs(wavefunctions[k, :]) ** 2)

    # 2. Normalize U by dividing each column by the square root of its norm
    wavefunctions_normalized = wavefunctions / np.sqrt(norms)
    # Return sorted real parts as requested
    return np.real(energies), wavefunctions_normalized  #


def solve_numerov_corrected(r_grid, V_grid, l):
    K = len(r_grid)
    delta = r_grid[1] - r_grid[0]
    alpha = 1  # Z=1 for Hydrogen atom

    # Effective potential including centrifugal term
    W = V_grid + (l * (l + 1)) / (2 * r_grid ** 2)

    H = np.zeros((K, K))
    N = np.zeros((K, K))

    inv_2delta2 = 1 / (2 * (delta ** 2))
    one_twelfth = 1 / 12
    ten_twelfths = 10 / 12

    # Standard matrix assembly
    for k in range(K):
        H[k, k] = (1 / (delta ** 2)) + (ten_twelfths * W[k])
        N[k, k] = ten_twelfths

        if k > 0:
            H[k, k - 1] = -inv_2delta2 + (one_twelfth * W[k - 1])
            N[k, k - 1] = one_twelfth
        if k < K - 1:
            H[k, k + 1] = -inv_2delta2 + (one_twelfth * W[k + 1])
            N[k, k + 1] = one_twelfth

    # Apply Task 3 correction for singularity at r=0

    H[0, 0] = H[0, 0] + (1 / 12) * (alpha / delta)

    # Solving the generalized eigenvalue problem H*u = E*N*u
    energies, wavefunctions = eigh(H, N)

    # Physical Normalization (Trapezoidal rule)
    norms = np.sqrt(np.sum(0.5 * delta * (wavefunctions[:-1, :] ** 2 + wavefunctions[1:, :] ** 2), axis=0))
    wavefunctions_normalized = wavefunctions / norms

    return np.real(energies), wavefunctions_normalized


def solve_numerov_corrected_order2(r_grid, V_grid, l):
    K = len(r_grid)
    delta = r_grid[1] - r_grid[0]
    alpha = 1  # Z=1 for Hydrogen atom

    # Effective potential including centrifugal term
    W = V_grid + (l * (l + 1)) / (2 * r_grid ** 2)

    H = np.zeros((K, K))
    N = np.zeros((K, K))

    inv_2delta2 = 1 / (2 * (delta ** 2))
    one_twelfth = 1 / 12
    ten_twelfths = 10 / 12

    # Standard matrix assembly
    for k in range(K):
        # Main diagonal
        H[k, k] = (1 / (delta ** 2)) + (ten_twelfths * W[k])
        N[k, k] = ten_twelfths

        # Off diagonals
        if k > 0:
            H[k, k - 1] = -inv_2delta2 + (one_twelfth * W[k - 1])
            N[k, k - 1] = one_twelfth
        if k < K - 1:
            H[k, k + 1] = -inv_2delta2 + (one_twelfth * W[k + 1])
            N[k, k + 1] = one_twelfth

    # --- TASK 4: Second Order Correction ---
    # The full correction includes:
    # 1. First order term (+1/12 * 1/delta) to handle the 1/r singularity.

    correction = (1 / 12) * (alpha / delta) + (1 / 30) * (alpha ** 2)

    # Apply correction to the first diagonal element
    H[0, 0] = H[0, 0] + correction

    # Solve the generalized eigenvalue problem H*u = E*N*u
    energies, wavefunctions = eigh(H, N)

    # Physical Normalization (Trapezoidal rule)
    norms = np.sqrt(np.sum(0.5 * delta * (wavefunctions[:-1, :] ** 2 + wavefunctions[1:, :] ** 2), axis=0))
    wavefunctions_normalized = wavefunctions / norms

    return np.real(energies), wavefunctions_normalized



##---------------Assignment 1------------:



def run_task_1_Harmonic_Oscillator():
    if not os.path.exists(harmonic_folder_path):
        os.makedirs(harmonic_folder_path)

    # Parameters
    K_list = [40, 80, 120, 240, 360, 480, 600]
    R = 10
    l = 0
    exact_energy = 1.5

    # Data Storage
    energies_fd_list = []
    energies_sn_list = []
    errors_fd = []
    errors_sn = []


    def analytical_u(r_grid):
        return (4 / np.pi ** 0.5) ** 0.5 * r_grid * np.exp(-r_grid ** 2 / 2)


    def power_law(K, C, q):
        return C * K ** (-q)


    # Main Loop
    for k in K_list:
        r = np.linspace(R / k, R, k)
        V_harmonic = 0.5 * r ** 2
        delta = r[1] - r[0]

        # Solve using original routines
        energies_fd, wavefunctions_normalized_fd = solve_finite_difference(r, V_harmonic, l)
        energies_sn, wavefunctions_normalized_sn = solve_numerov(r, V_harmonic, l)

        # Store energy values and errors
        energies_fd_list.append(energies_fd[0])
        energies_sn_list.append(energies_sn[0])
        errors_fd.append(abs(energies_fd[0] - exact_energy))
        errors_sn.append(abs(energies_sn[0] - exact_energy))

        # Extract Ground States (first column)
        ground_state_fd = wavefunctions_normalized_fd[:, 0]
        ground_state_sn = wavefunctions_normalized_sn[:, 0]

        # Fix signs to ensure positive peaks
        if np.mean(ground_state_fd) < 0: ground_state_fd *= -1
        if np.mean(ground_state_sn) < 0: ground_state_sn *= -1

        # Physical Normalization (Simpson Rule)
        norm_fd = np.sqrt(np.sum(0.5 * delta * (ground_state_fd[:-1] ** 2 + ground_state_fd[1:] ** 2)))
        u_fd = ground_state_fd / norm_fd

        norm_sn = np.sqrt(np.sum(0.5 * delta * (ground_state_sn[:-1] ** 2 + ground_state_sn[1:] ** 2)))
        u_sn = ground_state_sn / norm_sn

        # Plotting individual K comparison (Both methods together)
        plt.figure(figsize=(8, 4))
        plt.plot(r, u_fd, label=f'FD (K={k})')
        plt.plot(r, u_sn, '--', label=f'Numerov (K={k})')
        plt.title(f'Wavefunction comparison for K={k}')
        plt.xlabel('r')
        plt.ylabel('u(r)')
        plt.legend()
        plt.grid(True)

        # Save each K separately
        plt.savefig(os.path.join(harmonic_folder_path, f"K{k}.png"), dpi=300)
        plt.show()
        plt.close()

    # --- Special Plot for K=600 with Analytical Comparison ---
    r_600 = np.linspace(R / 600, R, 600)
    u_exact = analytical_u(r_600)

    plt.figure(figsize=(10, 6))
    # Using the results from the last iteration (K=600)
    plt.plot(r_600, u_fd, 'b-', label='Finite Difference (K=600)')
    plt.plot(r_600, u_sn, 'g--', label='Numerov (K=600)')
    plt.plot(r_600, u_exact, 'r:', label='Analytical Solution', linewidth=2)

    plt.title('Final Comparison at K=600 vs Analytical')
    plt.xlabel('r')
    plt.ylabel('u(r)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(harmonic_folder_path, "Final_Comparison_K600.png"), dpi=300)
    plt.show()

    # --- Log-Log Convergence Plot ---
    plt.figure(figsize=(8, 6))
    plt.loglog(K_list, errors_fd, 'ob', label='Finite Difference')
    plt.loglog(K_list, errors_sn, 'sg', label='Numerov')

    # Curve fitting to find q
    popt_fd, _ = curve_fit(power_law, K_list, errors_fd)
    popt_sn, _ = curve_fit(power_law, K_list, errors_sn)

    plt.loglog(K_list, power_law(np.array(K_list), *popt_fd), 'b--', label=f'Fit FD: q={popt_fd[1]:.2f}')
    plt.loglog(K_list, power_law(np.array(K_list), *popt_sn), 'g--', label=f'Fit SN: q={popt_sn[1]:.2f}')

    plt.title('Log-Log Convergence Plot')
    plt.xlabel('K')
    plt.ylabel('$\eta = |\epsilon - 1.5|$')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.savefig(os.path.join(harmonic_folder_path, "Convergence_LogLog.png"), dpi=300)
    plt.show()

    # --- Energy Table Print out ---
    print(f"{'K':<5} | {'FD Energy':<15} | {'Numerov Energy':<15}")
    print("-" * 40)
    for i, k in enumerate(K_list):
        print(f"{k:<5} | {energies_fd_list[i]:<15.10f} | {energies_sn_list[i]:<15.10f}")




##---------------Assignment 2------------:





def run_task_2_Hydrogen_Atom():
    if not os.path.exists(electric_folder_path):
        os.makedirs(electric_folder_path)

    # Parameters
    K_list = [80, 120, 240, 360, 480, 600]
    R = 50
    l = 2
    n = 3
    exact_energy = -1/(2*(n**2))

    # Data Storage
    energies_fd_list = []
    energies_sn_list = []
    errors_fd = []
    errors_sn = []


    def analytical_u_hydrogen_1_0(r_grid):
        # Analytical ground state for Hydrogen atom (n=1, l=0)
        return 2 * r_grid * np.exp(-r_grid)


    def power_law(K, C, q):
        return C * K ** (-q)


    # Main Loop
    for k in K_list:
        r = np.linspace(R / k, R, k)
        V_electric = -1 / r
        delta = r[1] - r[0]

        # Solve using original routines
        energies_fd, wavefunctions_normalized_fd = solve_finite_difference(r, V_electric, l)
        energies_sn, wavefunctions_normalized_sn = solve_numerov(r, V_electric, l)

        # Store energy values and errors
        energies_fd_list.append(energies_fd[0])
        energies_sn_list.append(energies_sn[0])
        errors_fd.append(abs(energies_fd[0] - exact_energy))
        errors_sn.append(abs(energies_sn[0] - exact_energy))

        # Extract Ground States (first column)
        ground_state_fd = wavefunctions_normalized_fd[:, 0]
        ground_state_sn = wavefunctions_normalized_sn[:, 0]

        # Fix signs to ensure positive peaks
        if np.mean(ground_state_fd) < 0: ground_state_fd *= -1
        if np.mean(ground_state_sn) < 0: ground_state_sn *= -1

        # Physical Normalization (Simpson Rule)
        norm_fd = np.sqrt(np.sum(0.5 * delta * (ground_state_fd[:-1] ** 2 + ground_state_fd[1:] ** 2)))
        u_fd = ground_state_fd / norm_fd

        norm_sn = np.sqrt(np.sum(0.5 * delta * (ground_state_sn[:-1] ** 2 + ground_state_sn[1:] ** 2)))
        u_sn = ground_state_sn / norm_sn

        # Plotting individual K comparison (Both methods together)
        plt.figure(figsize=(8, 4))
        plt.plot(r, u_fd, label=f'FD (K={k})')
        plt.plot(r, u_sn, '--', label=f'Numerov (K={k})')
        plt.title(f'Wavefunction comparison for K={k}')
        plt.xlabel('r')
        plt.ylabel('u(r)')
        plt.legend()
        plt.grid(True)

        # Save each K separately
        plt.savefig(os.path.join(electric_folder_path, f"K{k}_l={l}.png"), dpi=300)
        plt.show()
        plt.close()

    # --- Special Plot for K=600 with Analytical Comparison ---
    r_600 = np.linspace(R / 600, R, 600)
    u_exact = analytical_u_hydrogen_1_0(r_600)

    plt.figure(figsize=(10, 6))
    # Using the results from the last iteration (K=600)
    plt.plot(r_600, u_fd, 'b-', label='Finite Difference (K=600)')
    plt.plot(r_600, u_sn, 'g--', label='Numerov (K=600)')
    plt.plot(r_600, u_exact, 'r:', label='Analytical Solution', linewidth=2)

    plt.title(f'Final Comparison at K=600 vs Analytical l={l}')
    plt.xlabel('r')
    plt.ylabel('u(r)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(electric_folder_path, f"Final_Comparison_K600_l={l}.png"), dpi=300)
    plt.show()

    # --- Log-Log Convergence Plot ---
    plt.figure(figsize=(8, 6))
    plt.loglog(K_list, errors_fd, 'ob', label='Finite Difference')
    plt.loglog(K_list, errors_sn, 'sg', label='Numerov')

    # Curve fitting to find q
    popt_fd, _ = curve_fit(power_law, K_list, errors_fd)
    popt_sn, _ = curve_fit(power_law, K_list, errors_sn)

    plt.loglog(K_list, power_law(np.array(K_list), *popt_fd), 'b--', label=f'Fit FD: q={popt_fd[1]:.2f}')
    plt.loglog(K_list, power_law(np.array(K_list), *popt_sn), 'g--', label=f'Fit SN: q={popt_sn[1]:.2f}')

    plt.title(f'Log-Log Convergence Plot l={l}')
    plt.xlabel('K')
    plt.ylabel(f'$\eta = |\epsilon +{-exact_energy}|$')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.savefig(os.path.join(electric_folder_path, f"Convergence_LogLog_l={l}.png"), dpi=300)
    plt.show()

    # --- Energy Table Print out ---
    print(f"{'K':<5} | {'FD Energy':<15} | {'Numerov Energy':<15}")
    print("-" * 40)
    for i, k in enumerate(K_list):
        print(f"{k:<5} | {energies_fd_list[i]:<15.10f} | {energies_sn_list[i]:<15.10f}")

    if not os.path.exists(electric_folder_path):
        os.makedirs(electric_folder_path)




##---------------Assignment 3--------------:




def run_task_3_Corrected_Numerov_1():
    # Parameters for Task 3: Corrected Numerov
    K_list = [80, 120, 240, 360, 480, 600]
    R = 50
    l = 0  # Ground state check
    n = 1
    exact_energy = -0.5  # Exact energy for Hydrogen ground state

    # Storage for convergence data
    errors_sn_corrected = []


    def power_law(K, C, q):
        return C * K ** (-q)

    # --- Calculation Loop ---
    print("Running calculations for Corrected Numerov...")
    for k in K_list:
        delta = R / (k + 1)
        r = np.arange(1, k + 1) * delta  # The first point r[0] is exactly delta
        V_electric = -1 / r

        # Calculate energy using the corrected routine
        energies_sn, _ = solve_numerov_corrected(r, V_electric, l)

        # Store energy error relative to exact E=-0.5
        error = abs(energies_sn[0] - exact_energy)
        errors_sn_corrected.append(error)
        print(f"K={k}, Energy={energies_sn[0]:.10f}, Error={error:.2e}")

    # --- Generate Log-Log Convergence Plot ---
    plt.figure(figsize=(8, 6))
    plt.loglog(K_list, errors_sn_corrected, 'sg', label='Corrected Numerov Data 1')

    # Curve fitting to determine the modified convergence rate (q)
    popt_sn, _ = curve_fit(power_law, K_list, errors_sn_corrected)

    plt.loglog(K_list, power_law(np.array(K_list), *popt_sn), 'g--',
               label=f'Corrected SN Fit: q={popt_sn[1]:.2f}')

    plt.title(f'Log-Log Convergence Plot (Corrected Numerov order 1) l={l}')
    plt.xlabel('Number of grid points (K)')
    plt.ylabel(r'$\eta = |\epsilon + 0.5|$')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.savefig(os.path.join(electric_folder_path, "Task3_Convergence.png"), dpi=300)
    plt.show()

    print("-" * 30)
    print(f"Final q value (Task 3): {popt_sn[1]:.2f}")




##---------------Assignment 4--------------:




def run_task_4_Corrected_Numerov_2():
    # Parameters for Task 4: Corrected Numerov
    K_list = [80, 120, 240, 360, 480, 600]
    R = 50
    l = 0  # Ground state check
    n = 1
    exact_energy = -0.5  # Exact energy for Hydrogen ground state

    # Storage for convergence data
    errors_sn_corrected = []


    def power_law(K, C, q):
        return C * K ** (-q)

    # --- Calculation Loop ---
    print("Running calculations for Corrected Numerov...")
    for k in K_list:
        delta = R / (k + 1)
        r = np.arange(1, k + 1) * delta  # The first point r[0] is exactly delta
        V_electric = -1 / r

        # Calculate energy using the corrected routine
        energies_sn, _ = solve_numerov_corrected_order2(r, V_electric, l)

        # Store energy error relative to exact E=-0.5
        error = abs(energies_sn[0] - exact_energy)
        errors_sn_corrected.append(error)
        print(f"K={k}, Energy={energies_sn[0]:.10f}, Error={error:.2e}")

    # --- Generate Log-Log Convergence Plot ---
    plt.figure(figsize=(8, 6))
    plt.loglog(K_list, errors_sn_corrected, 'sg', label='Corrected Numerov Data 2')

    # Curve fitting to determine the modified convergence rate (q)
    popt_sn, _ = curve_fit(power_law, K_list, errors_sn_corrected)

    plt.loglog(K_list, power_law(np.array(K_list), *popt_sn), 'g--',
               label=f'Corrected SN Fit: q={popt_sn[1]:.2f}')

    plt.title(f'Log-Log Convergence Plot (Corrected Numerov order 2) l={l}')
    plt.xlabel('Number of grid points (K)')
    plt.ylabel(r'$\eta = |\epsilon + 0.5|$')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.savefig(os.path.join(electric_folder_path, "Task4_Convergence.png"), dpi=300)
    plt.show()

    print("-" * 30)
    print(f"Final q value (Task 3): {popt_sn[1]:.2f}")



#---------------Assignment 5------------:




def run_task_5_radius_scan():
    print("--- Running Task 5: Box Radius Dependence ---")

    # שימוש בפתרון המתוקן (Order 1 או 2 לבחירתך, כאן שמתי את המתוקן הבסיסי)
    # נגדיר פונקציה פנימית קטנה לנוחות או נשתמש במה שיש
    def get_energy_for_R(R, K=480, l=0):
        delta = R / (K + 1)
        r = np.arange(1, K + 1) * delta
        V = -1 / r
        # שימוש ב-solver הקיים שלך
        energies, _ = solve_numerov_corrected(r, V, l)
        return energies

    K_fixed = 480
    l = 0
    R_values = np.linspace(1, 100, 50)
    n_levels = 5
    errors_by_n = {n: [] for n in range(1, n_levels + 1)}

    for R in R_values:
        energies = get_energy_for_R(R, K=K_fixed, l=l)

        for n in range(1, n_levels + 1):
            exact_E = -0.5 / (n ** 2)
            if n - 1 < len(energies):
                calc_E = energies[n - 1]
                error = abs(calc_E - exact_E)
            else:
                error = np.nan
            errors_by_n[n].append(error)

    # Plotting
    plt.figure(figsize=(10, 7))
    for n in range(1, n_levels + 1):
        plt.semilogy(R_values, errors_by_n[n], 'o-', label=f'n={n}')

    plt.title(f'Task 5: Error vs Box Radius R (K={K_fixed}, l=0)')
    plt.xlabel('Radius R [a.u.]')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True, which="both", alpha=0.5)

    save_path = "Task5_Radius_Dependence.png"
    if os.path.exists(electric_folder_path):
        save_path = os.path.join(electric_folder_path, "Task5_Radius_Dependence.png")

    plt.savefig(save_path)
    plt.show()
    print("Task 5 Done.")






#---------------Assignment 6------------:




def get_electron_configuration(Z):
    """
    Returns a list of shells to fill for atomic number Z.
    Format: [(n, l, num_electrons), ...]
    Example for Z=4 (Be): [(1, 0, 2), (2, 0, 2)] -> 1s^2 2s^2
    """
    config = []
    electrons_left = Z

    # Order of filling: 1s, 2s, 2p, 3s, 3p...
    # (n, l, capacity)
    orbitals = [
        (1, 0, 2),  # 1s
        (2, 0, 2),  # 2s
        (2, 1, 6),  # 2p
        (3, 0, 2),  # 3s
        (3, 1, 6)  # 3p
    ]

    for n, l, cap in orbitals:
        if electrons_left <= 0:
            break
        take = min(electrons_left, cap)
        config.append({'n': n, 'l': l, 'count': take})
        electrons_left -= take

    return config


def calculate_V_ee(r_grid, rho):
    """
    Calculates the electron-electron potential from charge density rho.
    Using the electrostatic integral solution for spherical symmetry.
    V(r) = Integral of rho(r') / max(r, r')
    """
    V = np.zeros_like(r_grid)
    delta = r_grid[1] - r_grid[0]

    # We solve the integral:
    # Potential at r = (1/r)*Integral(0 to r) + Integral(r to inf)

    # Integral 1: Inner part (charge inside r) / r
    # Integral 2: Outer part (charge outside r) / r'

    # Using cumulative trapezoidal integration for efficiency
    integrand1 = rho * r_grid ** 2  # Charge density * 4pi*r^2 factor (4pi cancels later)
    integrand2 = rho * r_grid  # Charge density * r

    # Numerical integration
    integral_in = np.zeros_like(r_grid)
    integral_out = np.zeros_like(r_grid)

    # Cumulative sum (like cumtrapz)
    integral_in = np.cumsum(integrand1) * delta

    # Reverse cumulative sum for the outer part
    integral_out = np.cumsum(integrand2[::-1])[::-1] * delta

    # Avoid division by zero at r=0
    term1 = np.zeros_like(r_grid)
    term1[1:] = integral_in[1:] / r_grid[1:]
    term1[0] = term1[1]  # Approximation at origin

    term2 = integral_out

    # The factor 4pi comes from angular integration
    # Definition in text: rho = (1/4pi) sum |u|^2/r^2
    # So the 4pi cancels out with the volume element d^3r = 4pi r^2 dr
    # V_ee(r) = 4*pi * [ (1/r)*Int_0^r rho(x)x^2 dx + Int_r^inf rho(x)x dx ]

    V_ee = 4 * np.pi * (term1 + term2)
    return V_ee


# --- 3. Main SCF Loop ---

def run_task_6_radius_Helium(K=500, R=20, max_iter=10):
    Z = 2  # Helium
    delta = R / K
    r = np.linspace(delta, R, K)

    # Arrays to store history for plotting
    history_rho = []
    history_Vee = []

    # Initial Guess: V_ee = 0
    V_ee = np.zeros_like(r)

    print(f"Starting SCF for Helium (Z={Z})...")

    for iteration in range(max_iter):
        # A. Total Potential = Nucleus + V_ee
        V_nucleus = -Z / r
        V_total = V_nucleus + V_ee

        # B. Solve Schrodinger for l=0 (Helium ground state is 1s^2)
        energies, wavefunctions = solve_numerov_corrected(r, V_total, l=0)

        # Ground state is the first eigenstate (index 0)
        u_1s = wavefunctions[:, 0]
        E_1s = energies[0]

        # C. Calculate Density
        # rho = (1/4pi) * sum( |u|^2 / r^2 )
        # For He: 2 electrons in 1s orbital
        density_contribution = (u_1s ** 2) / (r ** 2)
        rho_new = (1 / (4 * np.pi)) * (2 * density_contribution)

        # D. Calculate New V_ee
        # Apply Fermi-Amaldi factor (Z-1)/Z = (2-1)/2 = 0.5
        factor = (Z - 1) / Z
        V_ee_calc = factor * calculate_V_ee(r, rho_new)

        # E. Mixing (Relaxation) step 5 in instructions
        V_ee_new = 0.5 * V_ee + 0.5 * V_ee_calc

        # Store for plotting
        history_rho.append(rho_new)
        history_Vee.append(V_ee)  # Store the V_ee used in this step

        # Check convergence
        diff = np.mean(np.abs(V_ee_new - V_ee))
        print(f"Iter {iteration + 1}: E_1s = {E_1s:.5f} Ha, V_diff = {diff:.2e}")

        V_ee = V_ee_new



    # 1. Density rho(r)
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 3, 1)
    for i in range(len(history_rho)):
        plt.plot(r, history_rho[i], label=f'Iter {i + 1}')
    plt.xlim(0, 5)  # Zoom in on the atom
    plt.title(r"Density $\rho(r)$ Convergence")
    plt.xlabel("r [a.u.]")
    plt.grid(True)
    plt.legend()

    # 2. Potential V_ee(r)
    plt.subplot(1, 3, 2)
    for i in range(len(history_Vee)):
        plt.plot(r, history_Vee[i], label=f'Iter {i + 1}')
    plt.xlim(0, 5)
    plt.title(r"Potential $V_{ee}(r)$ Convergence")
    plt.xlabel("r [a.u.]")
    plt.grid(True)
    plt.legend()
    plt.subplot(1, 3, 3)
    Z_eff = -Z + r * V_ee  # Using final V_ee
    plt.plot(r, Z_eff, 'k-', linewidth=2)
    plt.xlim(0, 5)
    plt.title(r"Effective Charge $Z_{eff}(r)$")
    plt.xlabel("r [a.u.]")
    plt.ylabel(r"$Z_{eff}$")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("Task6_Helium_SCF.png")
    plt.show()

    print(f"Final Ground State Energy (single electron parameter): {E_1s:.5f} Ha")
    print(
        f"Total Atom Energy (approx): {2 * E_1s - np.trapz(0.5 * rho_new * V_ee * 4 * np.pi * r ** 2, r):.5f} Ha (Note: simple sum is not total energy)")




#---------------Assignment 7------------:




def get_oxygen_config():
    """
    Oxygen configuration: 1s2, 2s2, 2p4
    Structure: (n, l, number_of_electrons)
    """
    return [
        {'n': 1, 'l': 0, 'count': 2},  # 1s
        {'n': 2, 'l': 0, 'count': 2},  # 2s
        {'n': 2, 'l': 1, 'count': 4}  # 2p
    ]




def run_task_7_radius_Oxygen(K=400, R=10, max_iter=40):
    Z = 8  # Oxygen
    delta = R / K
    r = np.linspace(delta, R, K)

    # Start with no repulsion
    V_ee = np.zeros_like(r)
    config = get_oxygen_config()

    # Lists to store history for plotting
    history_rho = []
    history_Vee = []
    plot_indices = []  # Keep track of which iterations we save

    print(f"Starting SCF for Oxygen (Z={Z})...")

    final_energies = {}  # Dictionary to store final energies

    for i in range(max_iter):
        V_nucleus = -Z / r
        V_total = V_nucleus + V_ee

        rho_new = np.zeros_like(r)

        # We need to solve the equation for every distinct l in the configuration
        # For Oxygen: l=0 (for 1s, 2s) and l=1 (for 2p)
        distinct_l = sorted(list(set(c['l'] for c in config)))

        current_iter_energies = {}

        for l in distinct_l:
            energies, wavefunctions = solve_numerov_corrected(r, V_total, l)


            for item in config:
                if item['l'] == l:
                    # Find the correct index in the solver output
                    # For l=0: n=1 is index 0, n=2 is index 1
                    # For l=1: n=2 is index 0 (lowest p state)
                    # General formula: index = n - (l + 1)
                    idx = item['n'] - (l + 1)

                    u_nl = wavefunctions[:, idx]
                    E_nl = energies[idx]
                    count = item['count']

                    # Add to total density
                    rho_new += (1 / (4 * np.pi)) * count * (u_nl ** 2) / (r ** 2)

                    # Store energy for printing
                    key = f"{item['n']}{'s' if l == 0 else 'p'}"
                    current_iter_energies[key] = E_nl

        # Calculate new potential and mixing
        factor = (Z - 1) / Z  # Fermi-Amaldi correction
        V_ee_calc = factor * calculate_V_ee(r, rho_new)

        V_ee_new = 0.5 * V_ee + 0.5 * V_ee_calc  # Mixing step

        # Save for plotting (only every 5 iterations to avoid clutter)
        if i % 5 == 0 or i == max_iter - 1:
            history_rho.append(rho_new)
            history_Vee.append(V_ee)
            plot_indices.append(i + 1)
            print(f"Iter {i + 1}: Energies = {current_iter_energies}")

        V_ee = V_ee_new
        final_energies = current_iter_energies

    # --- Generate Plots (Task 7) ---
    plt.figure(figsize=(14, 5))

    # 1. Density
    plt.subplot(1, 3, 1)
    for j, rho_plot in enumerate(history_rho):
        # Label only first and last to keep legend clean
        label = f'Iter {plot_indices[j]}' if j == 0 or j == len(history_rho) - 1 else None
        plt.plot(r, rho_plot, label=label)
    plt.xlim(0, 4)  # Zoom in
    plt.title("Electron Density Convergence")
    plt.xlabel("r [a.u.]")
    plt.legend()
    plt.grid(True)

    # 2. Repulsion Potential
    plt.subplot(1, 3, 2)
    for j, vee_plot in enumerate(history_Vee):
        plt.plot(r, vee_plot)
    plt.xlim(0, 4)
    plt.title("Repulsion Potential $V_{ee}$")
    plt.xlabel("r [a.u.]")
    plt.grid(True)

    # 3. Effective Charge (Final iteration only)
    plt.subplot(1, 3, 3)
    Z_eff = -Z + r * V_ee
    plt.plot(r, Z_eff, 'k-', linewidth=2)
    plt.xlim(0, 5)
    plt.title(r"Effective Charge $Z_{eff}(r)$")
    plt.xlabel("r [a.u.]")
    plt.ylabel(r"$Z_{eff}$")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("Task7_Oxygen_SCF.png")
    plt.show()

    print("\n--- Final Converged Energies (Hartree) ---")
    for orb, en in final_energies.items():
        print(f"Orbital {orb}: {en:.5f} Ha")


if __name__ == "__main__":

    run_task_1_Harmonic_Oscillator()
    #run_task_2_Hydrogen_Atom()
    #run_task_3_Corrected_Numerov_1()
    #run_task_4_Corrected_Numerov_2()
    #run_task_5_radius_scan()
    #run_task_6_radius_Helium(K=500, R=20)
    #run_task_7_radius_Oxygen(K=400, R=10)