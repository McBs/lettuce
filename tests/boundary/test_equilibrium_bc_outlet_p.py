from tests.conftest import *


# TODO: Implement cuda_native generator and test suite

def test_equilibrium_outlet_p_algorithm(fix_stencil, fix_configuration):
    """
    Test for the equilibrium outlet p boundary algorithm. This test verifies
    that the algorithm correctly computes the
    equilibrium outlet pressure by comparing its output to manually calculated
    equilibrium values.
    """
    device, dtype, use_native = fix_configuration
    if use_native:
        pytest.skip("This test does not depend on the native implementation.")
    context = Context(device=device, dtype=dtype, use_native=False)
    stencil = fix_stencil

    flow = TestFlow(context, stencil=stencil, resolution=16,
                    reynolds_number=1, mach_number=0.1)
    direction = [0] * (stencil.d - 1) + [1]
    boundary_cpu = EquilibriumOutletP(flow=flow, direction=direction,
                                      rho_outlet=1.2)
    f_post_boundary = boundary_cpu(flow)[..., -1]
    u_slice = [stencil.d, *flow.resolution[:stencil.d - 1], 1]
    rho_slice = [1, *flow.resolution[:stencil.d - 1], 1]
    u = flow.units.convert_velocity_to_lu(context.one_tensor(u_slice))
    rho = context.one_tensor(rho_slice) * 1.2
    reference = flow.equilibrium(flow, rho=rho, u=u)[..., 0]
    assert reference.cpu().numpy() == pytest.approx(
        f_post_boundary.cpu().numpy(), rel=1e-2)  # TODO rel is too big!
