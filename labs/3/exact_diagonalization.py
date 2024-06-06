import numpy as np

# define basic Pauli matrices
s_alpha = [np.array([[1, 0], [0, 1]], dtype='complex'),
           np.array([[0, 1], [1, 0]], dtype='complex'),
           np.array([[0, -1j], [1j, 0]],dtype='complex'),
           np.array([[1, 0], [0, -1]], dtype='complex')]


# define the many-body spin operators
def sp(alpha, n, N):
    Sa = s_alpha[alpha]
    for i in range(n):
        Sa = np.kron(s_alpha[0], Sa)
    for j in range(n+1, N):
        Sa = np.kron(Sa, s_alpha[0])
    return Sa


def magn_exact_diagonalization(N, g, t, Npoints):
    """
    Benchmark the Trotterized circuit evolution with the exact Hamiltonian time evolution,
    obtained through exact diagonalization (only for small system sizes N!). 
    This function returns the magnetization for equally spaced timesteps between  0  and  t .
    """
    # array containing the magnetization of individual basis states
    magnetization_basis_states = -np.array( [np.sum(2*np.array([int(bin(n)[2:].zfill(N)[i]) for i in range(N)]) - 1.0)/N for n in range(2**N)] )

    # create the hamiltonian
    hamiltonian = np.zeros((2**N, 2**N), dtype='complex')
    for i in range(N):
        hamiltonian += g/2*sp(1, i, N)
        if i != N-1:
            hamiltonian += -1/2*sp(3, i, N) @ sp(3, i+1, N)

    # diagonalize
    E, V = np.linalg.eig(hamiltonian)

    # time evolve
    magnetization = np.zeros(Npoints)
    initial_state = np.array([int(n==0) for n in range(2**N)])
    overlap = V.transpose().conj() @ initial_state
    for ind,T in enumerate(np.linspace(0,t,Npoints)):
        state_evolved = V @ (np.exp(-1j*T*E) * overlap)
        magnetization[ind] = np.sum(magnetization_basis_states * np.abs(state_evolved)**2)

    return magnetization


# Useful functions to verify whether a given density matrix is physical

def is_Hermitian(M, rtol = 1e-5, atol = 1e-9):
    return np.allclose(M, np.conjugate(M.T), rtol=rtol, atol=atol)

def is_positive(M, tol = 1e-7):
    s = np.linalg.eigvalsh(M)
    assert (s[0] > -tol)
    for i in range(len(s)):
      if s[i] <= 0:
         s[i] = 1e-12
    return s