#include "rte_kernel_launcher_cuda.h"

namespace
{
    #include "rte_solver_kernels.cu"
}

template<typename TF>
void lw_solver_noscat_gaussquad(
        const int ncol, const int nlay, const int ngpt, const BOOL_TYPE top_at_1, const int nmus,
        const TF* ds, const TF* weights, const TF* tau, const TF* lay_source,
        const TF* lev_source_inc, const TF* lev_source_dec, const TF* sfc_emis,
        const TF* sfc_src, TF* flux_up, TF* flux_dn,
        const TF* sfc_src_jac, TF* flux_up_jac)
{
    TF eps = std::numeric_limits<TF>::epsilon();

    const int sfc_size = ncol * ngpt;
    const int vol_size = sfc_size * nlay;
    const int flx_size = sfc_size * (nlay + 1);

    TF* source_sfc = Tools_gpu::allocate_gpu<TF>(sfc_size);
    TF* source_sfc_jac = Tools_gpu::allocate_gpu<TF>(sfc_size);
    TF* sfc_albedo = Tools_gpu::allocate_gpu<TF>(sfc_size);
    TF* tau_loc = Tools_gpu::allocate_gpu<TF>(vol_size);
    TF* trans = Tools_gpu::allocate_gpu<TF>(vol_size);
    TF* source_dn = Tools_gpu::allocate_gpu<TF>(vol_size);
    TF* source_up = Tools_gpu::allocate_gpu<TF>(vol_size);
    TF* radn_dn = Tools_gpu::allocate_gpu<TF>(flx_size);
    TF* radn_up = Tools_gpu::allocate_gpu<TF>(flx_size);
    TF* radn_up_jac = Tools_gpu::allocate_gpu<TF>(flx_size);


    // Running some permutations of block sizes.
    /*`
    {
        std::cout << "TUNING lw_solver_noscat_gaussquad_kernel" << std::endl;
        std::vector<std::pair<int, int>> col_gpt_combis;
        std::vector<int> cols{ 1, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
        std::vector<int> gpts{ 1, 2, 4, 8, 16, 32, 64, 128};
        for (const int igpt : gpts)
            for (const int icol : cols)
                col_gpt_combis.emplace_back(icol, igpt);

        // Create tmp arrays to write output to.
        Array_gpu<TF,3> flux_up_tmp{flux_up}, flux_dn_tmp{flux_dn}, flux_up_jac_tmp{flux_up_jac};

        for (const auto& p : col_gpt_combis)
        {
            std::cout << "(" << p.first << ", " << p.second << "): ";

            const int block_col2d = p.first;
            const int block_gpt2d = p.second;

            const int grid_col2d = ncol/block_col2d + (ncol%block_col2d > 0);
            const int grid_gpt2d = ngpt/block_gpt2d + (ngpt%block_gpt2d > 0);

            dim3 grid_gpu2d(grid_col2d, grid_gpt2d);
            dim3 block_gpu2d(block_col2d, block_gpt2d);

            // Warm it up.
            lw_solver_noscat_gaussquad_kernel<<<grid_gpu2d, block_gpu2d>>>(
                    ncol, nlay, ngpt, eps, top_at_1, nmus, ds, weights, tau, lay_source,
                    lev_source_inc, lev_source_dec, sfc_emis, sfc_src, radn_up,
                    radn_dn, sfc_src_jac, radn_up_jac, tau_loc, trans, source_dn, source_up,
                    source_sfc, sfc_albedo, source_sfc_jac, flux_up_tmp, flux_dn_tmp, flux_up_jac_tmp);

            cudaEvent_t start;
            cudaEvent_t stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            const int n_samples = 10;

            cudaEventRecord(start, 0);
            for (int i=0; i<n_samples; ++i)
                lw_solver_noscat_gaussquad_kernel<<<grid_gpu2d, block_gpu2d>>>(
                        ncol, nlay, ngpt, eps, top_at_1, nmus, ds, weights, tau, lay_source,
                        lev_source_inc, lev_source_dec, sfc_emis, sfc_src, radn_up,
                        radn_dn, sfc_src_jac, radn_up_jac, tau_loc, trans, source_dn, source_up,
                        source_sfc, sfc_albedo, source_sfc_jac, flux_up_tmp, flux_dn_tmp, flux_up_jac_tmp);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float duration = 0.f;
            cudaEventElapsedTime(&duration, start, stop);

            std::cout << std::setprecision(10) << duration << " (ns), check: " << flux_up_tmp({ncol, nlay+1, ngpt}) << ", ";

            // Check whether kernel has succeeded;
            cudaError err = cudaGetLastError();
            if (err != cudaSuccess)
                std::cout << cudaGetErrorString(err) << std::endl;
            else
                std::cout << std::endl;
        }

        std::cout << "STOP TUNING lw_solver_noscat_gaussquad_kernel" << std::endl;
    }
    */
    // End of performance tuning.

    const int block_col2d = 64;
    const int block_gpt2d = 2;

    const int grid_col2d = ncol/block_col2d + (ncol%block_col2d > 0);
    const int grid_gpt2d = ngpt/block_gpt2d + (ngpt%block_gpt2d > 0);

    dim3 grid_gpu2d(grid_col2d, grid_gpt2d);
    dim3 block_gpu2d(block_col2d, block_gpt2d);

    const int block_col3d = 96;
    const int block_lay3d = 1;
    const int block_gpt3d = 1;

    const int grid_col3d = ncol/block_col3d + (ncol%block_col3d > 0);
    const int grid_lay3d = (nlay+1)/block_lay3d + ((nlay+1)%block_lay3d > 0);
    const int grid_gpt3d = ngpt/block_gpt3d + (ngpt%block_gpt3d > 0);

    dim3 grid_gpu3d(grid_col3d, grid_lay3d, grid_gpt3d);
    dim3 block_gpu3d(block_col3d, block_lay3d, block_gpt3d);

    const int top_level = top_at_1 ? 0 : nlay;

    lw_solver_noscat_step1_kernel<<<grid_gpu3d, block_gpu3d>>>(
            ncol, nlay, ngpt, eps, top_at_1, ds, weights, tau, lay_source,
            lev_source_inc, lev_source_dec, sfc_emis, sfc_src, flux_up, flux_dn, sfc_src_jac,
            flux_up_jac, tau_loc, trans, source_dn, source_up, source_sfc, sfc_albedo, source_sfc_jac);

    lw_solver_noscat_step2_kernel<<<grid_gpu2d, block_gpu2d>>>(
            ncol, nlay, ngpt, eps, top_at_1, ds, weights, tau, lay_source,
            lev_source_inc, lev_source_dec, sfc_emis, sfc_src, flux_up, flux_dn, sfc_src_jac,
            flux_up_jac, tau_loc, trans, source_dn, source_up, source_sfc, sfc_albedo, source_sfc_jac);
        
    lw_solver_noscat_step3_kernel<<<grid_gpu3d, block_gpu3d>>>(
            ncol, nlay, ngpt, eps, top_at_1, ds, weights, tau, lay_source,
            lev_source_inc, lev_source_dec, sfc_emis, sfc_src, flux_up, flux_dn, sfc_src_jac,
            flux_up_jac, tau_loc, trans, source_dn, source_up, source_sfc, sfc_albedo, source_sfc_jac);
        
    apply_BC_kernel_lw<<<grid_gpu2d, block_gpu2d>>>(top_level, ncol, nlay, ngpt, top_at_1, flux_dn, radn_dn);

    if (nmus > 1)
    {
        for (int imu=1; imu<nmus; ++imu)
        {
            lw_solver_noscat_step1_kernel<<<grid_gpu3d, block_gpu3d>>>(
                    ncol, nlay, ngpt, eps, top_at_1, ds+imu, weights+imu, tau, lay_source,
                    lev_source_inc, lev_source_dec, sfc_emis, sfc_src, radn_up, radn_dn, sfc_src_jac,
                    radn_up_jac, tau_loc, trans, source_dn, source_up, source_sfc, sfc_albedo, source_sfc_jac);

            lw_solver_noscat_step2_kernel<<<grid_gpu2d, block_gpu2d>>>(
                    ncol, nlay, ngpt, eps, top_at_1, ds+imu, weights+imu, tau, lay_source,
                    lev_source_inc, lev_source_dec, sfc_emis, sfc_src, radn_up, radn_dn, sfc_src_jac,
                    radn_up_jac, tau_loc, trans, source_dn, source_up, source_sfc, sfc_albedo, source_sfc_jac);

            lw_solver_noscat_step3_kernel<<<grid_gpu3d, block_gpu3d>>>(
                    ncol, nlay, ngpt, eps, top_at_1, ds+imu, weights+imu, tau, lay_source,
                    lev_source_inc, lev_source_dec, sfc_emis, sfc_src, radn_up, radn_dn, sfc_src_jac,
                    radn_up_jac, tau_loc, trans, source_dn, source_up, source_sfc, sfc_albedo, source_sfc_jac);

            add_fluxes_kernel<<<grid_gpu3d, block_gpu3d>>>(
                    ncol, nlay+1, ngpt,
                    radn_up, radn_dn, radn_up_jac,
                    flux_up, flux_dn, flux_up_jac);
        }
    }

    Tools_gpu::free_gpu<TF>(source_sfc);
    Tools_gpu::free_gpu<TF>(source_sfc_jac);
    Tools_gpu::free_gpu<TF>(sfc_albedo);
    Tools_gpu::free_gpu<TF>(tau_loc);
    Tools_gpu::free_gpu<TF>(trans);
    Tools_gpu::free_gpu<TF>(source_dn);
    Tools_gpu::free_gpu<TF>(source_up);
    Tools_gpu::free_gpu<TF>(radn_dn);
    Tools_gpu::free_gpu<TF>(radn_up);
    Tools_gpu::free_gpu<TF>(radn_up_jac);
}


extern "C" 
{
    void lw_solver_noscat_gaussquad_wrapper_(const int* ncol, const int* nlay, const int* ngpt, const BOOL_TYPE* top_at_1, const int* nmus,
    const double* ds, const double* weights, const double* tau, const double* lay_source,
    const double* lev_source_inc, const double* lev_source_dec, const double* sfc_emis,
    const double* sfc_src, double* flux_up, double* flux_dn,
    const double* sfc_src_jac, double* flux_up_jac)
    {
        lw_solver_noscat_gaussquad(*ncol, *nlay, *ngpt, *top_at_1, *nmus,
                                ds, weights, tau, lay_source, 
                                lev_source_inc, lev_source_dec, sfc_emis, sfc_src,
                                flux_up, flux_dn, sfc_src_jac, flux_up_jac);
    }
}
