# Runs the "Examples" code in README.md, writes the printed output back into
# each code block (as "#>" lines), and redraws the figure that follows each
# block in docs/figures/. Mirrors data-raw/readme_figures.R in the R package
# GICHighDimension. The run times exclude compilation: the hidden helper block
# in the README compiles every method once before anything is timed.
#
# Run from the package root:
#   julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'   # first time only
#   julia --project=docs docs/readme_figures.jl
#
# Only the code from the "Examples" heading onward is run, so the installation
# code is skipped. As in the R README, a statement's value is shown unless the
# statement is an assignment (including `.=` and `+=`), a definition, a
# `using`, a call to a mutating function (one whose name ends in `!`, such as
# `Random.seed!`), or ends with `;`.

using Plots
include(joinpath(@__DIR__, "glmnet_patch.jl"))   # GLMNet.jl fix used by the lasso examples

# Search methods (lines) and the lasso fits (squares in the model-size panel)
const PALETTE = ["AIC" => "#2a78d6", "BIC" => "#eb6834", "GIC6" => "#1baf7a",
                 "Boltzmann" => "#d0459b"]
const LASSO = ["Lasso" => "#8a5cc2", "AdaLasso" => "#b8860b"]
const METHODS = vcat(PALETTE, LASSO)
const SIZE_STYLE = ["n1000" => :solid, "n2000" => :dot]   # solid: n = 1000, dotted: n = 2000

# Lighter shade of a colour (used for n = 1000)
function tint(hex)
    rgb = [parse(Int, hex[i:i+1], base = 16) for i in (2, 4, 6)]
    RGB(((rgb .+ (255 .- rgb) .* 0.55) ./ 255)...)
end

fit_of(fits, s, m) = getfield(getfield(fits, Symbol(s)), Symbol(m))

# Three panels: each criterion's value, and its model size, after every search
# step, with one line per criterion and sample size; then the run time of every
# method, including the lasso and adaptive lasso. Light grey vertical lines
# mark a new pass over the predictors; the dashed line marks the true size.
# `fits` is fits.<sample size>.<method>, as returned by compare_criteria().
function plot_criteria(fits, p, true_size, file)
    gr()
    nsteps = length(fit_of(fits, "n1000", "AIC").values)
    steps  = 1:nsteps
    passes = p:p:(nsteps - 1)
    path_sizes  = [length.(fit_of(fits, s, m).models) for (s, _) in SIZE_STYLE for (m, _) in PALETTE]
    lasso_sizes = [length(fit_of(fits, s, m).selected) for (s, _) in SIZE_STYLE for (m, _) in LASSO]
    log_y = maximum(maximum, path_sizes) > 100
    log_ticks = sort(unique([5, 10, 100, 500, true_size]))

    common = (legend = false, grid = false, framestyle = :box, titlefontsize = 11,
              guidefontsize = 10, tickfontsize = 9, left_margin = 6Plots.mm,
              bottom_margin = 3Plots.mm)
    crit = plot(; title = "Criterion along the search", xlabel = "Search step",
                ylabel = "Criterion value", common...)
    size_ylabel = log_y ? "Number of predictors (log scale)" : "Number of predictors"
    msize = plot(; title = "Model size (true size = $true_size)", xlabel = "Search step",
                 ylabel = size_ylabel, common...)
    if log_y
        lo, hi = extrema(vcat(log_ticks, reduce(vcat, path_sizes), lasso_sizes))
        plot!(msize; yscale = :log10, yticks = (log_ticks, string.(log_ticks)),
              ylims = (lo / 1.15, hi * 1.15))
    end
    for pl in (crit, msize)
        isempty(passes) || vline!(pl, collect(passes); color = "#d9d9d9", lw = 0.8)
    end
    hline!(msize, [true_size]; color = "#595959", ls = :dash, lw = 1)

    for (s, style) in SIZE_STYLE, (m, color) in PALETTE
        fit = fit_of(fits, s, m)
        plot!(crit, steps, fit.values; seriestype = :steppost, color = color, ls = style, lw = 2)
        plot!(msize, steps, length.(fit.models); seriestype = :steppost, color = color,
              ls = style, lw = 2)
    end
    # Lasso and adaptive lasso have no search path: their final model sizes are
    # squares at the right edge (light: n = 1000, dark: n = 2000)
    for (i, (s, _)) in enumerate(SIZE_STYLE), (m, color) in LASSO
        x = nsteps * (i == 1 ? 0.94 : 0.98)
        scatter!(msize, [x], [length(fit_of(fits, s, m).selected)]; marker = :rect,
                 markersize = 7, markerstrokewidth = 0,
                 color = i == 1 ? tint(color) : color)
    end

    # Run time of every method: light bars n = 1000, dark bars n = 2000
    secs = [max(fit_of(fits, s, m).seconds, 0.005) for (m, _) in METHODS, (s, _) in SIZE_STYLE]
    ylo, yhi = 10.0^floor(log10(minimum(secs))), 10.0^ceil(log10(maximum(secs)))
    ticks = filter(t -> ylo <= t <= yhi, [m * 10.0^e for e in -3:3 for m in (1, 2, 5)])
    ticklabels = [t >= 1 ? string(round(Int, t)) : rstrip(string(round(t, sigdigits = 1)), '0')
                  for t in ticks]
    rt = plot(; title = "Run time",
              ylabel = "Seconds (log scale)", yscale = :log10,
              ylims = (ylo, yhi), yticks = (ticks, ticklabels),
              xlims = (0.4, length(METHODS) + 0.6),
              xticks = (1:length(METHODS), first.(METHODS)), xrotation = 30, common...)
    annotate!(rt, 0.55, yhi / 1.6, text("light: n = 1000\ndark: n = 2000", 8, :left, :top))
    for (i, (m, color)) in enumerate(METHODS), j in 1:2
        x0 = i - 0.36 + (j - 1) * 0.36
        plot!(rt, Shape([x0, x0 + 0.34, x0 + 0.34, x0], [ylo, ylo, secs[i, j], secs[i, j]]);
              color = j == 1 ? tint(color) : color, linewidth = 0)
    end

    # Legend rows under the panels: method colours, then sample-size line types
    legend_row(ncol) = plot(; framestyle = :none, legend = :top, legend_columns = ncol,
                            foreground_color_legend = nothing,
                            background_color_legend = nothing, legendfontsize = 9,
                            xlims = (0, 1), ylims = (0, 1))
    key1 = legend_row(length(METHODS))
    for (m, color) in PALETTE
        plot!(key1, [NaN], [NaN]; color = color, lw = 2.5, label = m)
    end
    for (m, color) in LASSO
        scatter!(key1, [NaN], [NaN]; color = color, marker = :rect, markersize = 6,
                 markerstrokewidth = 0, label = m)
    end
    key2 = legend_row(2)
    for (s, style) in SIZE_STYLE
        plot!(key2, [NaN], [NaN]; color = "#404040", ls = style, lw = 2,
              label = "n = " * s[2:end])
    end

    fig = plot(crit, msize, rt, key1, key2;
               layout = @layout([a b c; d{0.06h}; e{0.06h}]),
               size = (1150, 470), dpi = 200)
    mkpath(dirname(file))
    savefig(fig, file)
    println("wrote ", file)
end

# Should the value of `ex` be shown, as at the REPL with R's visibility rules?
function visible(ex, src)
    endswith(rstrip(src), ";") && return false
    ex isa Expr || return true
    ex.head == :toplevel && return visible(ex.args[end], src)   # a; b on one line
    ex.head in (:(=), :function, :using, :import, :struct, :const, :macro, :for,
                :while, :module, :abstract, :global) && return false
    if ex.head == :macrocall && length(ex.args) >= 3 && ex.args[end] isa Expr
        return visible(ex.args[end], src)            # e.g. @time x = f()
    end
    endswith(string(ex.head), "=") && return false   # .=, +=, ...
    if ex.head == :call
        f = ex.args[1]
        name = f isa Symbol ? f : (f isa Expr && f.head == :. ? f.args[end] : nothing)
        name isa QuoteNode && (name = name.value)
        name isa Symbol && endswith(string(name), "!") && return false
    end
    return true
end

function show_value(value)
    io = IOBuffer()
    # invokelatest: show methods defined by the README code itself
    Base.invokelatest(show, IOContext(io, :limit => true, :displaysize => (40, 92)),
                      MIME"text/plain"(), value)
    split(String(take!(io)), '\n')
end

# Evaluate one code block in `mod` and return its lines with fresh output.
function run_block(lines, mod)
    lines = filter(l -> !startswith(l, "#>"), lines)
    text = join(lines, '\n')
    out = [String[] for _ in lines]
    pos = 1
    while true
        # Skip blank and comment-only lines, so they are not counted as part of
        # the next statement
        while pos <= ncodeunits(text)
            m = match(r"\A(?:[ \t]*(?:#[^\n]*)?\n)", SubString(text, pos))
            m === nothing && break
            pos += ncodeunits(m.match)
        end
        pos > ncodeunits(text) && break
        ex, next = Meta.parse(text, pos; greedy = true)
        ex === nothing && break
        # The statement's source, without trailing blank and comment-only lines
        src = replace(text[pos:prevind(text, next)], r"(\n[ \t]*(#[^\n]*)?)*\s*\z" => "")
        last = count(==('\n'), SubString(text, 1, prevind(text, pos))) + 1 +
               count(==('\n'), src)
        ex isa Expr && ex.head == :error && error("parse error in README block: ", ex.args[1])
        value = Core.eval(mod, ex)
        if value !== nothing && visible(ex, src)
            append!(out[last], "#> " .* rstrip.(show_value(value)))
        end
        pos = next
    end
    vcat([vcat(l, o) for (l, o) in zip(lines, out)]...)
end

function process(path, start_heading, figure_dir)
    doc = readlines(path)
    i = findfirst(l -> occursin(start_heading, l), doc)
    mod = Module(:ReadmeExamples)
    Core.eval(mod, :(using GICVariableSelection))
    while i <= length(doc)
        if occursin(r"^```julia\s*$", doc[i])
            stop  = i + findfirst(l -> occursin(r"^```\s*$", l), doc[(i + 1):end])
            block = run_block(doc[(i + 1):(stop - 1)], mod)
            doc   = vcat(doc[1:i], block, doc[stop:end])
            i    += length(block) + 1

            nxt = i + 1
            while nxt <= length(doc) && !startswith(doc[nxt], "```")
                m = match(r"README-[a-z-]+\.png", doc[nxt])
                if m !== nothing
                    multi = occursin("multivariate", m.match)
                    truth = getfield(mod, multi ? :true_union : :true_cols)
                    plot_criteria(getfield(mod, :fits), getfield(mod, :p), length(truth),
                                  joinpath(figure_dir, m.match))
                    break
                end
                nxt += 1
            end
        end
        i += 1
    end
    open(path, "w") do io
        foreach(l -> println(io, l), doc)
    end
    println("updated ", path)
end

process("README.md", "## Examples", joinpath("docs", "figures"))
