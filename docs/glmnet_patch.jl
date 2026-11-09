# Fix for GLMNet.jl 0.7.4: for the multi-response Gaussian (MvNormal) and
# Multinomial families it returns the coefficients as `ca[sortperm(ia), :, :]`,
# which sorts the packed coefficient rows but does not place them at their
# predictors' rows. Row j of `betas` is then predictor j only if every
# predictor entered the path; otherwise the coefficients land on the wrong
# predictors (and predict() and glmnetcv() use the wrong ones too). This
# reloads the two affected source files with row k of `ca` scattered to
# predictor ia[k].

using GLMNet

Core.eval(GLMNet, quote
    function _unpack_multi(ca, ia, nin, nvars, lmu)
        betas = zeros(Float64, nvars, size(ca, 2), lmu)
        for k in 1:maximum(nin[1:lmu]; init = 0)
            betas[ia[k], :, :] = ca[k, :, 1:lmu]
        end
        betas
    end
end)

let src = joinpath(pkgdir(GLMNet), "src")
    for file in ("Multinomial.jl", "MultiResponseNet.jl")
        code = read(joinpath(src, file), String)
        bug = "ca[sortperm(ia), :, 1:lmu]"
        occursin(bug, code) || error("GLMNet.jl has changed; check whether this patch is still needed")
        code = replace(code, bug => "_unpack_multi(ca, ia, nin, nvars, lmu)")
        redirect_stderr(devnull) do        # silence "method overwritten" warnings
            include_string(GLMNet, code, joinpath(src, file))
        end
    end
end
