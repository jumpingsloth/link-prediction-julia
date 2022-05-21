using LinearAlgebra

function set_diagonal(M::Matrix, val::Number)
    X = M
    cols = size(M, 1)
    rows = size(M, 2)
    for i in 1:cols, j in 1:rows
        if (i == j)
            X[i, j] = val
        end
    end

    return X
end

function gen_sym_adj_mat(n)
    A = rand(0.0:1.0, n, n)
    A = A - tril(A) + triu(A)'
    A = set_diagonal(A, 1)
    return A
end

### generate network
const num_of_layers = 3
const matrix_dimensions = 4
N = Matrix[]
append!(N, [gen_sym_adj_mat(matrix_dimensions) for i in 1:num_of_layers])
for A in N
    display(A)
    println()
end

### select target layer m in N (e.g. layer 2)
m = N[2]

# connection similarity between target layer m and auxiliary layer k
function θ(l::Matrix, m::Matrix)
    sum = 0.0
    for i in 1:matrix_dimensions
        numerator = 0.0
        denominator = 0.0

        for j in 1:matrix_dimensions
            numerator += l[i, j] * m[i, j]

            denominator += m[i, j]
            denominator += l[i, j]
            denominator -= l[i, j] * m[i, j]
        end

        sum += numerator / denominator
    end

    return (1 / size(N, 1)#= TODO: check if correct =#) * sum
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

# (norm = frobenius norm)

### compute proximity matrix WAm (proximity / similarity of each layer to m)
WAm = zeros(matrix_dimensions, matrix_dimensions)
for k in N
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
Cm = zeros(matrix_dimensions, matrix_dimensions)
CAm = rand(matrix_dimensions, matrix_dimensions)
CAm_old = nothing
Cm_old = nothing

println("\nApproximating Cm and CAm")
for i in 1:10 # choosen randomly
    global Cm_old = Cm
    global CAm_old = CAm

    # update steps
    global Cm = calc_c(WmT, Wm, WAm, CAm)
    global CAm = calc_c_a(WAmT, WAm, Wm, Cm)
    ΔCAm = norm(CAm_old - CAm)
    ΔCm = norm(Cm_old - Cm)

    println("i = $i\tΔCAm = $ΔCAm\tΔCm = $ΔCm")
end

### compute intralayer likelihood matrix Lm
Lm = Wm * Cm

### compute intralayer likelihood matrix Lm
LAm = WAm * CAm

println("\nIntralayer likelihood Lm:")
display(Lm)
println()

println("Interlayer likelihood LAm:")
display(LAm)
println()


