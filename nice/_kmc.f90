!! Copyright (C) 2020 Ayers Lab.
!!
!! This file is part of NICE.
!!
!! NICE is free software; you can redistribute it and/or modify it under
!! the terms of the GNU General Public License as published by the Free
!! Software Foundation; either version 3 of the License, or (at your
!! option) any later version.
!!
!! NICE is distributed in the hope that it will be useful, but WITHOUT
!! ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
!! FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
!! for more details.
!!
!! You should have received a copy of the GNU General Public License
!! along with this program; if not, see <http://www.gnu.org/licenses/>.


!! KMC SUBROUTINE
!!
subroutine run_kmc(nspc, nrxn, conc, stoich, f_consts, r_consts, f_rates, r_rates, n_rates, step, maxiter, time)
    !f2py intent(in) nspc, nrxn, stoich, f_consts, r_consts, step, maxiter
    !f2py intent(inout) conc, f_rates, r_rates, n_rates
    !f2py intent(out) time
    !f2py depend(nspc) conc
    !f2py depend(nrxn) f_consts, r_consts, f_rates, r_rates, n_rates
    !f2py depend(nspc,nrxn) stoich
    implicit none
    integer, parameter :: dp = kind(1.0d0)
    !! Arguments
    integer, intent(in) :: nspc, nrxn, maxiter
    real(kind=dp), intent(in) :: stoich(nspc, nrxn), f_consts(nrxn), r_consts(nrxn), step
    real(kind=dp), intent(inout) :: conc(nspc)
    real(kind=dp), intent(out) :: f_rates(nrxn), r_rates(nrxn), n_rates(nrxn), time
    !! Working arrays
    real(kind=dp), allocatable :: pvec(:)
    !! Internal variables
    integer :: i, j, n
    !! Allocate arrays
    n = nrxn * 2
    allocate(pvec(n))
    !! Run iterations
    call random_seed()
    time = 0.0d0
    do i = 1, maxiter
        call update_rates(nspc, nrxn, conc, stoich, f_consts, r_consts, f_rates, r_rates, n_rates)
        call select_reaction(nrxn, f_rates, r_rates, n, pvec, j)
        call do_reaction(nspc, nrxn, conc, stoich, f_rates, r_rates, step, j, time)
    end do
    !! Deallocate work arrays
    deallocate(pvec)

contains

subroutine select_reaction(nrxn, f_rates, r_rates, n, pvec, idx)
    implicit none
    integer, parameter :: dp = kind(1.0d0)
    !! Arguments
    integer, intent(in) :: nrxn, n
    integer, intent(out) :: idx
    real(kind=dp), intent(in) :: f_rates(nrxn), r_rates(nrxn)
    real(kind=dp), intent(inout) :: pvec(n)
    !! Internal variables
    integer :: i
    real(kind=dp) :: t, u
    !! Construct probability vector
    t = 0.0d0
    do i = 1, nrxn
        t = t + f_rates(i)
        pvec(i) = t
    end do
    do i = 1, nrxn
        t = t + r_rates(i)
        pvec(i + nrxn) = t
    end do
    !! Select random reaction
    call random_number(u)
    t = t * u
    do i = 1, n
        if (pvec(i) .gt. t) then
            idx = i
            return
        end if
    end do
    idx = n
end subroutine !! select_reaction

subroutine do_reaction(nspc, nrxn, conc, stoich, f_rates, r_rates, step, idx, time)
    implicit none
    !! Arguments
    integer, intent(in) :: nspc, nrxn, idx
    real(kind=dp), intent(in) :: stoich(nspc, nrxn), f_rates(nrxn), r_rates(nrxn), step
    real(kind=dp), intent(inout) :: conc(nspc), time
    !! Internal variables
    integer :: j, k
    real(kind=dp) :: r
    !! Do reaction
    if (idx .le. nrxn) then
        r = f_rates(idx)
        if (r .gt. 0.0d0) then
            do j = 1, nspc
                conc(j) = conc(j) + stoich(j, idx) * step
            end do
            time = time + step / r
        end if
    else
        k = idx - nrxn
        r = r_rates(k)
        if (r .gt. 0.0d0) then
            do j = 1, nspc
                conc(j) = conc(j) - stoich(j, k) * step
            end do
            time = time + step / r
        end if
    end if
end subroutine !! do_reaction

end subroutine !! run_kmc


!! NEKMC SUBROUTINE
!!
subroutine run_nekmc(nspc, nrxn, conc, stoich, f_consts, r_consts, f_rates, r_rates, n_rates, step, maxiter, time)
    !f2py intent(in) nspc, nrxn, stoich, f_consts, r_consts, step, maxiter
    !f2py intent(inout) conc, f_rates, r_rates, n_rates
    !f2py intent(out) time
    !f2py depend(nspc) conc
    !f2py depend(nrxn) f_consts, r_consts, f_rates, r_rates, n_rates
    !f2py depend(nspc,nrxn) stoich
    implicit none
    integer, parameter :: dp = kind(1.0d0)
    !! Arguments
    integer, intent(in) :: nspc, nrxn, maxiter
    real(kind=dp), intent(in) :: stoich(nspc, nrxn), f_consts(nrxn), r_consts(nrxn), step
    real(kind=dp), intent(inout) :: conc(nspc)
    real(kind=dp), intent(out) :: f_rates(nrxn), r_rates(nrxn), n_rates(nrxn), time
    !! Working arrays
    real(kind=dp), allocatable :: pvec(:)
    !! Internal variables
    integer :: i, j
    !! Allocate arrays
    allocate(pvec(nrxn))
    !! Run iterations
    call random_seed()
    time = 0.0d0
    do i = 1, maxiter
        call update_rates(nspc, nrxn, conc, stoich, f_consts, r_consts, f_rates, r_rates, n_rates)
        call select_reaction(nrxn, n_rates, pvec, j)
        call do_reaction(nspc, nrxn, conc, stoich, n_rates, step, j, time)
    end do
    !! Deallocate work arrays
    deallocate(pvec)

contains

subroutine select_reaction(nrxn, n_rates, pvec, idx)
    implicit none
    integer, parameter :: dp = kind(1.0d0)
    !! Arguments
    integer, intent(in) :: nrxn
    integer, intent(out) :: idx
    real(kind=dp), intent(in) :: n_rates(nrxn)
    real(kind=dp), intent(inout) :: pvec(nrxn)
    !! Internal variables
    integer :: i
    real(kind=dp) :: t, u
    !! Construct probability vector
    t = 0.0d0
    do i = 1, nrxn
        t = t + abs(n_rates(i))
        pvec(i) = t
    end do
    !! Select random reaction
    call random_number(u)
    t = t * u
    do i = 1, nrxn
        if (pvec(i) .gt. t) then
            idx = i
            return
        end if
    end do
    idx = nrxn
end subroutine !! select_reaction

subroutine do_reaction(nspc, nrxn, conc, stoich, n_rates, step, idx, time)
    implicit none
    integer, parameter :: dp = kind(1.0d0)
    !! Arguments
    integer, intent(in) :: nspc, nrxn, idx
    real(kind=dp), intent(in) :: stoich(nspc, nrxn), n_rates(nrxn), step
    real(kind=dp), intent(inout) :: conc(nspc), time
    !! Internal variables
    integer :: j
    real(kind=dp) :: r
    !! Do reaction
    r = n_rates(idx)
    if (r .gt. 0.0d0) then
        do j = 1, nspc
            conc(j) = conc(j) + stoich(j, idx) * step
        end do
        time = time + step / r
    else if (r .lt. 0.0d0) then
        do j = 1, nspc
            conc(j) = conc(j) - stoich(j, idx) * step
        end do
        time = time - step / r
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
    integer, parameter :: dp = kind(1.0d0)
    !! Arguments
    integer, intent(in) :: nspc, nrxn
    real(kind=dp), intent(in) :: conc(nspc), stoich(nspc, nrxn), f_consts(nrxn), r_consts(nrxn)
    real(kind=dp), intent(out) :: f_rates(nrxn), r_rates(nrxn), n_rates(nrxn)
    !! Internal variables
    integer :: i, j
    real(kind=dp) :: s, t, u
    !! Update rates
    do i = 1, nrxn
        t = f_consts(i)
        u = r_consts(i)
        do j = 1, nspc
            s = stoich(j, i)
            if (s .lt. 0.0d0) then
                t = t * conc(j) ** abs(s)
            else if (s .ge. 0.0d0) then
                u = u * conc(j) ** s
            end if
        end do
        f_rates(i) = t
        r_rates(i) = u
        n_rates(i) = t - u
    end do
end subroutine !! update_rates
