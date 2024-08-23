from qutip import *
import numpy as np
import qutip.core.data as _data


def steadyoper_floquet(H_0, c_ops, Op_t, w_d=1.0, n_it=3, sparse=False,
                        solver=None, **kwargs):
    """
    Calculates the effective steady state for a driven
     system with a time-dependent cosinusoidal term:

    .. math::

        \\mathcal{\\hat{H}}(t) = \\hat{H}_0 +
         \\mathcal{\\hat{O}} \\cos(\\omega_d t)

    Parameters
    ----------
    H_0 : :obj:`.Qobj`
        A Hamiltonian or Liouvillian operator.

    c_ops : list
        A list of collapse operators.

    Op_t : :obj:`.Qobj`
        The the interaction operator which is multiplied by the cosine

    w_d : float, default: 1.0
        The frequency of the drive

    n_it : int, default: 3
        The number of iterations for the solver

    sparse : bool, default: False
        Solve for the steady state using sparse algorithms.

    solver : str, optional
        Solver to use when solving the linear system.
        Default supported solver are:

        - "solve", "lstsq"
          dense solver from numpy.linalg
        - "spsolve", "gmres", "lgmres", "bicgstab"
          sparse solver from scipy.sparse.linalg
        - "mkl_spsolve"
          sparse solver by mkl.

        Extensions to qutip, such as qutip-tensorflow, may provide their own
        solvers. When ``H_0`` and ``c_ops`` use these data backends, see their
        documentation for the names and details of additional solvers they may
        provide.

    **kwargs:
        Extra options to pass to the linear system solver. See the
        documentation of the used solver in ``numpy.linalg`` or
        ``scipy.sparse.linalg`` to see what extra arguments are supported.

    Returns
    -------
    dm : qobj
        Steady state density matrix.

    Notes
    -----
    See: Sze Meng Tan,
    https://painterlab.caltech.edu/wp-content/uploads/2019/06/qe_quantum_optics_toolbox.pdf,
    Section (16)

    """
    if H_0.type == 'oper':
        L_0 = liouvillian(H_0, c_ops)
    else:
        if H_0.type == 'super':
            for i in range(0, len(c_ops)):
                L_0 = H_0 + sum(c_ops)
    L_m = 0.5 * liouvillian(Op_t)
    L_p = 0.5 * liouvillian(Op_t)
    # L_p and L_m correspond to the positive and negative
    # frequency terms respectively.
    # They are independent in the model, so we keep both names.
    Id = qeye_like(L_0)
    S = qzero_like(L_0)
    T = qzero_like(L_0)

    if isinstance(H_0.data, _data.CSR) and not sparse:
        L_0 = L_0.to("Dense")
        L_m = L_m.to("Dense")
        L_p = L_p.to("Dense")
        Id = Id.to("Dense")

    for n_i in np.arange(n_it, 0, -1):
        L = L_0 - 1j * n_i * w_d * Id + L_m @ S
        S.data = - _data.solve(L.data, L_p.data, solver, kwargs)
        L = L_0 + 1j * n_i * w_d * Id + L_p @ T
        T.data = - _data.solve(L.data, L_m.data, solver, kwargs)

    M_subs = L_0 + L_m @ S + L_p @ T
    return steadystate(M_subs, solver=solver, **kwargs)