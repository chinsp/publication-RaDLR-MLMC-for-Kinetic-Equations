__precompile__

using SparseArrays

struct UpwindStencil3D
    D⁺₁::SparseMatrixCSC{Float64, Int64}
    D⁺₂::SparseMatrixCSC{Float64, Int64}
    D⁺₃::SparseMatrixCSC{Float64, Int64}
    D⁻₁::SparseMatrixCSC{Float64, Int64}
    D⁻₂::SparseMatrixCSC{Float64, Int64}
    D⁻₃::SparseMatrixCSC{Float64, Int64}

    function UpwindStencil3D(settings::Settings, order::Int=2)
        nx, ny, nz = settings.NCellsX, settings.NCellsY, settings.NCellsZ
        Δx, Δy, Δz = settings.dx, settings.dy, settings.dz
        #density = settings.density
        density = ones(size(settings.density))

        D⁺₁ = spzeros(nx*ny*nz, nx*ny*nz)
        D⁺₂ = spzeros(nx*ny*nz, nx*ny*nz)
        D⁺₃ = spzeros(nx*ny*nz, nx*ny*nz)
        D⁻₁ = spzeros(nx*ny*nz, nx*ny*nz)
        D⁻₂ = spzeros(nx*ny*nz, nx*ny*nz)
        D⁻₃ = spzeros(nx*ny*nz, nx*ny*nz)

        if order == 1
            # First-order accuracy
            # Set up D⁺₁
            II = zeros(2*(nx-2)*(ny-2)*(nz-2)); J = zeros(2*(nx-2)*(ny-2)*(nz-2)); vals = zeros(2*(nx-2)*(ny-2)*(nz-2))
            counter = -1

            for i in 2:nx-1, j in 2:ny-1, k in 2:nz-1
                counter += 2

                # x part
                index = vectorIndex(nx, ny, i, j, k)
                indexMinus = vectorIndex(nx, ny, i-1, j, k)

                II[counter+1] = index
                J[counter+1] = index
                vals[counter+1] = 1 / Δx / density[i, j, k]
                if i > 1
                    II[counter] = index
                    J[counter] = indexMinus
                    vals[counter] = -1 / Δx / density[i-1, j, k]
                end
            end
            D⁺₁ = sparse(II, J, vals, nx*ny*nz, nx*ny*nz)

            # Set up D⁺₂
            II = zeros(2*(nx-2)*(ny-2)*(nz-2)); J = zeros(2*(nx-2)*(ny-2)*(nz-2)); vals = zeros(2*(nx-2)*(ny-2)*(nz-2))
            counter = -1

            for i in 2:nx-1, j in 2:ny-1, k in 2:nz-1
                counter += 2
                
                # y part
                index = vectorIndex(nx, ny, i, j, k)
                indexMinus = vectorIndex(nx, ny, i, j-1, k)
                II[counter+1] = index
                J[counter+1] = index
                vals[counter+1] = 1 / Δy / density[i, j, k]
                if j > 1
                    II[counter] = index
                    J[counter] = indexMinus
                    vals[counter] = -1 / Δy / density[i, j-1, k]
                end
            end
            D⁺₂ = sparse(II, J, vals, nx*ny*nz, nx*ny*nz)
            
            # Set up D⁺₃
            II = zeros(2*(nx-2)*(ny-2)*(nz-2)); J = zeros(2*(nx-2)*(ny-2)*(nz-2)); vals = zeros(2*(nx-2)*(ny-2)*(nz-2))
            counter = -1

            for i in 2:nx-1, j in 2:ny-1, k in 2:nz-1
                counter += 2
                
                # z part
                index = vectorIndex(nx, ny, i, j, k)
                indexMinus = vectorIndex(nx, ny, i, j, k-1)
                II[counter+1] = index
                J[counter+1] = index
                vals[counter+1] = 1 / Δz / density[i, j, k]
                if k > 1
                    II[counter] = index
                    J[counter] = indexMinus
                    vals[counter] = -1 / Δz / density[i, j, k-1]
                end
            end
            D⁺₃ = sparse(II, J, vals, nx*ny*nz, nx*ny*nz)

            # Set up D⁻₁
            II = zeros(2*(nx-2)*(ny-2)*(nz-2)); J = zeros(2*(nx-2)*(ny-2)*(nz-2)); vals = zeros(2*(nx-2)*(ny-2)*(nz-2))
            counter = -1

            for i in 2:nx-1, j in 2:ny-1, k in 2:nz-1
                counter += 2

                # x part
                index = vectorIndex(nx,ny,i,j,k);
                indexPlus = vectorIndex(nx,ny,i+1,j,k);

                II[counter+1] = index;
                J[counter+1] = index;
                vals[counter+1] = -1/Δx / density[i,j,k]; 
                if i < nx
                    II[counter] = index;
                    J[counter] = indexPlus;
                    vals[counter] = 1/Δx / density[i+1,j,k]; 
                end
            end
            D⁻₁ = sparse(II, J, vals, nx*ny*nz, nx*ny*nz)

            # Set up D⁻₂
            II = zeros(2*(nx-2)*(ny-2)*(nz-2)); J = zeros(2*(nx-2)*(ny-2)*(nz-2)); vals = zeros(2*(nx-2)*(ny-2)*(nz-2))
            counter = -1

            for i in 2:nx-1, j in 2:ny-1, k in 2:nz-1
                counter += 2

                # y part
                index = vectorIndex(nx,ny,i,j,k);
                indexPlus = vectorIndex(nx,ny,i,j+1,k);

                II[counter+1] = index;
                J[counter+1] = index;
                vals[counter+1] = -1/Δy / density[i,j,k]; 
                if i < ny
                    II[counter] = index;
                    J[counter] = indexPlus;
                    vals[counter] = 1/Δy / density[i,j+1,k]; 
                end
            end
            D⁻₂ = sparse(II, J, vals, nx*ny*nz, nx*ny*nz)

            # Set up D⁻₃ 
            II = zeros(2*(nx-2)*(ny-2)*(nz-2)); J = zeros(2*(nx-2)*(ny-2)*(nz-2)); vals = zeros(2*(nx-2)*(ny-2)*(nz-2))
            counter = -1

            for i in 2:nx-1, j in 2:ny-1, k in 2:nz-1
                counter += 2

                 # z part
                index = vectorIndex(nx,ny,i,j,k);
                indexPlus = vectorIndex(nx,ny,i,j,k+1);

                II[counter+1] = index;
                J[counter+1] = index;
                vals[counter+1] = -1/Δz / density[i,j,k]; 
                if i < ny
                    II[counter] = index;
                    J[counter] = indexPlus;
                    vals[counter] = 1/Δz / density[i,j,k+1]; 
                end
            end
            D⁻₃ = sparse(II, J, vals, nx*ny*nz, nx*ny*nz)
            
        elseif order == 2
            # Second-order accuracy
            II = zeros(3*(nx-4)*(ny-4)*(nz-4)); J = zeros(3*(nx-4)*(ny-4)*(nz-4)); vals = zeros(3*(nx-4)*(ny-4)*(nz-4))
            # Set up D⁺₁
            Threads.@threads for i in 3:nx-2
                Threads.@threads for j in 3:ny-2
                    Threads.@threads for k in 3:nz-2
                        # x part
                        index = vectorIndex(nx, ny, i, j, k)
                        indexM = vectorIndex(nx, ny, i-1, j, k)
                        indexMM = vectorIndex(nx, ny, i-2, j, k)

                        II[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1] = index
                        J[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1] = index
                        vals[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1] = 3 / Δx / 2 / density[i, j, k]
                        II[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1+1] = index
                        J[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1+1] = indexM
                        vals[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1+1] = -4 / Δx / 2 / density[i-1, j, k]
                        II[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1+2] = index
                        J[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1+2] = indexMM
                        vals[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1+2] = 1 / Δx / 2 / density[i-2, j, k]
                    end 
                end
            end
            D⁺₁ = sparse(II, J, vals, nx*ny*nz, nx*ny*nz)
            
            II = zeros(3*(nx-4)*(ny-4)*(nz-4)); J = zeros(3*(nx-4)*(ny-4)*(nz-4)); vals = zeros(3*(nx-4)*(ny-4)*(nz-4))
            # Set up D⁺₂
            Threads.@threads for i in 3:nx-2
                Threads.@threads for j in 3:ny-2
                    Threads.@threads for k in 3:nz-2 
                         # y part
                        index = vectorIndex(nx, ny, i, j, k)
                        indexM = vectorIndex(nx, ny, i, j-1, k)
                        indexMM = vectorIndex(nx, ny, i, j-2, k)

                        II[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1] = index
                        J[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1] = index
                        vals[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1] = 3 / Δy / 2 / density[i, j, k]
                        II[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1+1] = index
                        J[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1+1] = indexM
                        vals[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1+1] = -4 / Δy / 2 / density[i, j-1, k]
                        II[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1+2] = index
                        J[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1+2] = indexMM
                        vals[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1+2] = 1 / Δy / 2 / density[i, j-2, k]
                    end
                end
            end
            D⁺₂ = sparse(II, J, vals, nx*ny*nz, nx*ny*nz)
            
            II = zeros(3*(nx-4)*(ny-4)*(nz-4)); J = zeros(3*(nx-4)*(ny-4)*(nz-4)); vals = zeros(3*(nx-4)*(ny-4)*(nz-4))
            counter = -2

            # Set up D⁺₃
            Threads.@threads for i in 3:nx-2
                Threads.@threads for j in 3:ny-2
                    Threads.@threads for k in 3:nz-2
                        # z part
                        index = vectorIndex(nx, ny, i, j, k)
                        indexM = vectorIndex(nx, ny, i, j, k-1)
                        indexMM = vectorIndex(nx, ny, i, j, k-2)

                        II[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1] = index
                        J[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1] = index
                        vals[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1] = 3 / Δz / 2 / density[i, j, k]
                        II[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1+1] = index
                        J[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1+1] = indexM
                        vals[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1+1] = -4 / Δz / 2 / density[i, j, k-1]
                        II[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1+2] = index
                        J[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1+2] = indexMM
                        vals[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1+2] = 1 / Δz / 2 / density[i, j, k-2]
                    end
                end
            end
            D⁺₃ = sparse(II, J, vals, nx*ny*nz, nx*ny*nz)

            II = zeros(3*(nx-4)*(ny-4)*(nz-4)); J = zeros(3*(nx-4)*(ny-4)*(nz-4)); vals = zeros(3*(nx-4)*(ny-4)*(nz-4))
            # Set up D⁻₁
            Threads.@threads for i in 3:nx-2
                Threads.@threads for j in 3:ny-2
                    Threads.@threads for k in 3:nz-2
                        # x part
                        index = vectorIndex(nx, ny, i, j, k)
                        indexM = vectorIndex(nx, ny, i+1, j, k)
                        indexMM = vectorIndex(nx, ny, i+2, j, k)

                        II[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1] = index
                        J[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1] = index
                        vals[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1] = -3 / Δx / 2 / density[i, j, k]
                        II[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1+1] = index
                        J[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1+1] = indexM
                        vals[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1+1] = 4 / Δx / 2 / density[i+1, j, k]
                        II[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1+2] = index
                        J[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1+2] = indexMM
                        vals[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1+2] = -1 / Δx / 2 / density[i+2, j, k]
                    end
                end
            end
            D⁻₁ = sparse(II, J, vals, nx*ny*nz, nx*ny*nz)
            
            counter = -2;
            II = zeros(3*(nx-4)*(ny-4)*(nz-4)); J = zeros(3*(nx-4)*(ny-4)*(nz-4)); vals = zeros(3*(nx-4)*(ny-4)*(nz-4))
            # Set up D⁻₂ 
            Threads.@threads for i in 3:nx-2
                Threads.@threads for j in 3:ny-2
                    Threads.@threads for k in 3:nz-2
                        # y part
                        index = vectorIndex(nx, ny, i, j, k)
                        indexM = vectorIndex(nx, ny, i, j+1, k)
                        indexMM = vectorIndex(nx, ny, i, j+2, k)

                        II[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1] = index
                        J[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1] = index
                        vals[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1] = -3 / Δy / 2 / density[i, j, k]
                        II[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1+1] = index
                        J[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1+1] = indexM
                        vals[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1+1] = 4 / Δy / 2 / density[i, j+1, k]
                        II[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1+2] = index
                        J[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1+2] = indexMM
                        vals[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1+2] = -1 / Δy / 2 / density[i, j+2, k]
                    end
                end
            end
            D⁻₂ = sparse(II, J, vals, nx*ny*nz, nx*ny*nz)

            counter = -2;
            II = zeros(3*(nx-4)*(ny-4)*(nz-4)); J = zeros(3*(nx-4)*(ny-4)*(nz-4)); vals = zeros(3*(nx-4)*(ny-4)*(nz-4))
            # Set up D⁻₃ 
            Threads.@threads for i in 3:nx-2
                Threads.@threads for j in 3:ny-2
                    Threads.@threads for k in 3:nz-2
                        # z part
                        index = vectorIndex(nx, ny, i, j, k)
                        indexM = vectorIndex(nx, ny, i, j, k+1)
                        indexMM = vectorIndex(nx, ny, i, j, k+2)

                        II[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1] = index
                        J[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1] = index
                        vals[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1] = -3 / Δz / 2 / density[i, j, k]
                        II[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1+1] = index
                        J[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1+1] = indexM
                        vals[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1+1] = 4 / Δz / 2 / density[i, j, k+1]
                        II[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1+2] = index
                        J[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1+2] = indexMM
                        vals[3*((i-3)*(ny-4)*(nz-4)+(j-3)*(nz-4)+(k-3))+1+2] = -1 / Δz / 2 / density[i, j, k+2]
                    end
                end
            end
            D⁻₃ = sparse(II, J, vals, nx*ny*nz, nx*ny*nz)

        end

        new(D⁺₁, D⁺₂, D⁺₃, D⁻₁, D⁻₂, D⁻₃)
    end
end
