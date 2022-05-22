using LinearAlgebra
using GraphRecipes, Plots
# using GraphPlot
using Crayons

function set_diagonal(M::Matrix, val::Number)
    X = M
    rows = size(M, 1)
    cols = size(M, 2)
    for i in 1:rows, j in 1:cols
        if (i == j)
            X[i, j] = val
        end
    end

    return X
end

function gen_sym_adj_mat(n)
    A = rand(0.0:1.0, n, n)
    A = A - tril(A) + triu(A)'
    A = set_diagonal(A, 0)
    return A
end

function mat_is_empty(m)
    empty = true;
    for i in 1:size(m, 1), j in 1:size(m, 2)
        if (m[i][j] != 0.0 && m[i][j] !== nothing)
            empty = false
        end
    end
    return empty
end


### generate network
const NUM_OF_LAYERS = 3
const MATRIX_SIZE = 4
G = Matrix[]
for i in 1:NUM_OF_LAYERS
    mat = gen_sym_adj_mat(MATRIX_SIZE)
    while mat_is_empty(mat)
        mat = gen_sym_adj_mat(MATRIX_SIZE)
    end
    push(G, mat)
end

for A in G
    display(A)
    println()
end

### select target layer m in N (e.g. layer 2)
m = G[2]

# connection similarity between target layer m and auxiliary layer k
function θ(l::Matrix, m::Matrix)
    sum = 0.0
    for i in 1:MATRIX_SIZE
        numerator = 0.0
        denominator = 0.0

        for j in 1:MATRIX_SIZE
            numerator += l[i, j] * m[i, j]

            denominator += m[i, j]
            denominator += l[i, j]
            denominator -= l[i, j] * m[i, j]
        end

        sum += numerator / denominator
    end
    N = size(G, 1)
    return (1 / N) * sum
end

function calc_c(WmT, Wm, WAm, CAm, α = 0.5, β = 0.5)
    # Cm = numpy.matmul(numpy.linalg.inv(numpy.matmul(WmT, Wm) + beta * numpy.eye(Wm.shape[0]) + alpha * numpy.matmul(WmT, Wm)), (numpy.matmul(WmT, Wm) + alpha * numpy.matmul(numpy.matmul(WmT, WAm), CAm)))
    Cm = inv(WmT*Wm + β*I + α*WmT*Wm) * (WmT*Wm + α*WmT*WAm*CAm)
    return Cm
end

function calc_c_a(WAmT, WAm, Wm, Cm, α = 0.5, γ = 0.5)
    # CAm = numpy.matmul(numpy.linalg.inv(numpy.matmul(WAmT, WAm) + gamma * numpy.eye(Wm.shape[0]) + alpha * numpy.matmul(WAmT, WAm)), (numpy.matmul(WAmT, WAm) + alpha * numpy.matmul(numpy.matmul(WAmT, Wm), Cm)))
    CAm = inv(WAmT*WAm + γ*I + α*WAmT*WAm) * (WAmT*WAm + α*WAmT*Wm*Cm)
    return CAm
end

#===============================================#

### compute proximity matrix WAm (proximity / similarity of each layer to m)
WAm = zeros(MATRIX_SIZE, MATRIX_SIZE)
for k in G
    # Wk = Ak but could be replaced
    Wk = k

    if (k != m)
        global WAm += θ(m, k) * Wk
    end
end

Wm = m
WmT = Wm'
WAmT = WAm'

display(WAm)
println()

### approximate intralayer contribution matrix Cm and interlayer contribution matrix CAm
Cm = zeros(MATRIX_SIZE, MATRIX_SIZE)
CAm = rand(MATRIX_SIZE, MATRIX_SIZE)
CAm_old = nothing
Cm_old = nothing

println("\nApproximating Cm and CAm")
for i in 1:10 # choosen randomly
    global Cm_old = Cm
    global CAm_old = CAm

    # update steps
    global Cm = calc_c(WmT, Wm, WAm, CAm)
    global CAm = calc_c_a(WAmT, WAm, Wm, Cm)

    # (norm = frobenius norm)
    ΔCAm = norm(CAm_old - CAm)
    ΔCm = norm(Cm_old - Cm)

    println("i = $i\tΔCAm = $ΔCAm\tΔCm = $ΔCm")
end

### compute intralayer likelihood matrix Lm
Lm = Wm * Cm

### compute intralayer likelihood matrix Lm
LAm = WAm * CAm


#==================================================#
# plot matrices

for i in 1:NUM_OF_LAYERS
    savefig(graphplot(G[i], names=1:MATRIX_SIZE), "G_$(i)_mat.png")
end

function float_to_uint(x)
    if (x < 0.0)
        return 0
    end

    return floor(UInt32, x)
end


function display_colored(m::Matrix)
    for i in 1:size(m, 1)
        for j in 1:size(m, 2)
            col = round(255 * m[i][j])
            if col < 0
                col = 0
            elseif col > 255
                col = 255
            end
            col = convert(UInt8, col)
            c = Crayon(background = (0,col,0), foreground = 0xFFFFFF)
            print(c, string(round(m[x][y], digits = 5)))
        end
        println()
    end
end

display(Lm)
println()
println("\nIntralayer likelihood Lm:")
display_colored(Lm)
println()

println("Interlayer likelihood LAm:")
display_colored(LAm)
println()
