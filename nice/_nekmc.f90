!! NEKMC SUBROUTINE
!!
subroutine run_nekmc(nspc, nrxn, conc, stoich, f_consts, r_consts, f_rates, r_rates, n_rates, step, maxiter)
    !f2py intent(in) nspc, nrxn, stoich, f_consts, r_consts, step, maxiter
    !f2py intent(inout) conc, f_rates, r_rates, n_rates
    !f2py depend(nspc) conc
    !f2py depend(nrxn) f_consts, r_consts, f_rates, r_rates, n_rates
    !f2py depend(nspc,nrxn) stoich
    implicit none
    !! Arguments
    integer(kind=8), intent(in) :: nspc, nrxn, maxiter
    real(kind=8), intent(in) :: stoich(nspc, nrxn), f_consts(nrxn), r_consts(nrxn), step
    real(kind=8), intent(inout) :: conc(nspc)
    real(kind=8), intent(out) :: f_rates(nrxn), r_rates(nrxn), n_rates(nrxn)
    !! Working arrays
    real(kind=8), allocatable :: pvec(:)
    !! Internal variables
    integer(kind=8) :: i, j
    !! Allocate arrays
    allocate(pvec(nrxn))
    !! Run iterations
    j = 0
    do i = 1, maxiter
        call update_rates(nspc, nrxn, conc, stoich, f_consts, r_consts, f_rates, r_rates, n_rates)
        call select_reaction(nrxn, n_rates, pvec, j)
        call do_reaction(nspc, nrxn, conc, stoich, n_rates, step, j)
    end do
    !! Deallocate work arrays
    deallocate(pvec)

contains

subroutine select_reaction(nrxn, n_rates, pvec, idx)
    implicit none
    !! Arguments
    integer(kind=8), intent(in) :: nrxn
    integer(kind=8), intent(out) :: idx
    real(kind=8), intent(in) :: n_rates(nrxn)
    real(kind=8), intent(inout) :: pvec(nrxn)
    !! Internal variables
    integer(kind=8) :: i
    real(kind=8) :: t
    !! Construct probability vector
    t = 0.0
    do i = 1, nrxn
        t = t + abs(n_rates(i))
        pvec(i) = t
    end do
    !! Select random reaction
    t = t * rand()
    do i = 1, nrxn
        if (pvec(i) .gt. t) then
            idx = i
            return
        end if
    end do
    idx = nrxn
end subroutine !! select_reaction

subroutine do_reaction(nspc, nrxn, conc, stoich, n_rates, step, idx)
    implicit none
    !! Arguments
    integer(kind=8), intent(in) :: nspc, nrxn, idx
    real(kind=8), intent(in) :: stoich(nspc, nrxn), n_rates(nrxn), step
    real(kind=8), intent(inout) :: conc(nspc)
    !! Internal variables
    integer(kind=8) :: j
    !! Do reaction
    if (n_rates(idx) .ge. 0.0) then
        do j = 1, nspc
            conc(j) = conc(j) + stoich(j, idx) * step
        end do
    else
        do j = 1, nspc
            conc(j) = conc(j) - stoich(j, idx) * step
        end do
    end if
end subroutine !! do_reaction

end subroutine !! run_nekmc


!! RATE UPDATE SUBROUTINE
!!
subroutine update_rates(nspc, nrxn, conc, stoich, f_consts, r_consts, f_rates, r_rates, n_rates)
    !f2py intent(in) nspc, nrxn, conc, stoich, f_consts, r_consts
    !f2py intent(inout) f_rates, r_rates, n_rates
    !f2py depend(nspc) conc
    !f2py depend(nrxn) f_consts, r_consts, f_rates, r_rates, n_rates
    !f2py depend(nspc,nrxn) stoich
    implicit none
    !! Arguments
    integer(kind=8), intent(in) :: nspc, nrxn
    real(kind=8), intent(in) :: conc(nspc), stoich(nspc, nrxn), f_consts(nrxn), r_consts(nrxn)
    real(kind=8), intent(out) :: f_rates(nrxn), r_rates(nrxn), n_rates(nrxn)
    !! Internal variables
    integer(kind=8) :: i, j
    real(kind=8) :: s, t, u
    !! Update rates
    do i = 1, nrxn
        t = f_consts(i)
        u = r_consts(i)
        do j = 1, nspc
            s = stoich(j, i)
            if (s .lt. 0.0) then
                t = t * conc(j) ** abs(s)
            else if (s .gt. 0.0) then
                u = u * conc(j) ** s
            end if
        end do
        f_rates(i) = t
        r_rates(i) = u
        n_rates(i) = t - u
    end do
end subroutine !! update_rates
