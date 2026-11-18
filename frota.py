"""
Global Solution 2026.2 - Modelagem Matemática e Computacional
Citizen Assist AI Mobilidade - Modelo da frota de ônibus

Etapa 1: reconstrução da distribuição atual da frota (sistema linear).
Etapa 2: política de redistribuição (transformação linear que preserva o total).

Execução:  python frota.py
Requisitos: Python 3.10+ e numpy (os cálculos principais usam frações exatas).
"""

from fractions import Fraction
from itertools import product
from math import ceil

import numpy as np

REGIOES = ["Norte", "Sul", "Leste", "Oeste"]
SIGLAS = ["N", "S", "L", "O"]

# ---------------------------------------------------------------------------
# Dados do Relatório I (registros operacionais)
# Variáveis: x = (N, S, L, O) = número de ônibus em cada região
# ---------------------------------------------------------------------------
#   N +  S +  L +  O = 72   (frota total)
#   N + 2S      +  O = 76   (Norte + Oeste + 2*Sul)
#        S + 2L      = 52   (Sul + 2*Leste)
#   N      +  L + 2O = 76   (Norte + Leste + 2*Oeste)
A = [
    [1, 1, 1, 1],
    [1, 2, 0, 1],
    [0, 1, 2, 0],
    [1, 0, 1, 2],
]
b = [72, 76, 52, 76]
DESCRICAO_EQUACOES = [
    "Frota total",
    "Norte + Oeste + 2*Sul",
    "Sul + 2*Leste",
    "Norte + Leste + 2*Oeste",
]

# ---------------------------------------------------------------------------
# Dados do Relatório II (demanda no horário de pico)
# ---------------------------------------------------------------------------
PASSAGEIROS = [560, 760, 720, 840]
CAPACIDADE = [40, 40, 40, 40]

# Vizinhança geográfica usada como critério de custo na redistribuição.
# Norte faz divisa com Leste e Oeste; Sul faz divisa com Leste e Oeste.
# Norte-Sul e Leste-Oeste são regiões opostas (deslocamento mais longo).
VIZINHAS = {("N", "L"), ("N", "O"), ("S", "L"), ("S", "O")}



def pct(v):
    """Percentual com uma casa decimal e vírgula (padrão brasileiro)."""
    return f"{v:.1f}".replace(".", ",")

def custo_deslocamento(origem: str, destino: str) -> int:
    """1 para regiões vizinhas, 2 para regiões opostas."""
    par = tuple(sorted((origem, destino)))
    return 1 if par in {tuple(sorted(p)) for p in VIZINHAS} else 2


# ---------------------------------------------------------------------------
# Utilidades de impressão
# ---------------------------------------------------------------------------
def fmt(v) -> str:
    if isinstance(v, Fraction):
        return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"
    return str(v)


def imprimir_matriz(M, titulo=None, larg=7):
    if titulo:
        print(titulo)
    for linha in M:
        print("  [" + " ".join(fmt(v).rjust(larg) for v in linha) + " ]")


def secao(titulo):
    print("\n" + "=" * 72)
    print(titulo)
    print("=" * 72)


# ---------------------------------------------------------------------------
# Álgebra linear com frações exatas
# ---------------------------------------------------------------------------
def determinante(M) -> Fraction:
    """Determinante por eliminação de Gauss (frações exatas)."""
    M = [[Fraction(v) for v in linha] for linha in M]
    n = len(M)
    det = Fraction(1)
    for c in range(n):
        piv = next((r for r in range(c, n) if M[r][c] != 0), None)
        if piv is None:
            return Fraction(0)
        if piv != c:
            M[c], M[piv] = M[piv], M[c]
            det = -det
        det *= M[c][c]
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            M[r] = [M[r][k] - f * M[c][k] for k in range(n)]
    return det


def posto(M) -> int:
    """Posto (rank) por escalonamento com frações exatas."""
    M = [[Fraction(v) for v in linha] for linha in M]
    linhas, colunas = len(M), len(M[0])
    r = 0
    for c in range(colunas):
        piv = next((i for i in range(r, linhas) if M[i][c] != 0), None)
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        for i in range(linhas):
            if i != r and M[i][c] != 0:
                f = M[i][c] / M[r][c]
                M[i] = [M[i][k] - f * M[r][k] for k in range(colunas)]
        r += 1
        if r == linhas:
            break
    return r


def gauss_jordan(A, b, verbose=True):
    """Resolve Ax = b por Gauss-Jordan, mostrando cada operação elementar."""
    n = len(A)
    M = [[Fraction(v) for v in A[i]] + [Fraction(b[i])] for i in range(n)]
    passos = []
    if verbose:
        imprimir_matriz(M, "Matriz aumentada inicial [A | b]:")
    for c in range(n):
        piv = next((r for r in range(c, n) if M[r][c] != 0), None)
        if piv is None:
            raise ValueError("Sistema sem pivô: não há solução única.")
        if piv != c:
            M[c], M[piv] = M[piv], M[c]
            passos.append(f"L{c+1} <-> L{piv+1}")
            if verbose:
                imprimir_matriz(M, f"\nTroca L{c+1} <-> L{piv+1}:")
        if M[c][c] != 1:
            p = M[c][c]
            M[c] = [v / p for v in M[c]]
            passos.append(f"L{c+1} <- L{c+1} / {fmt(p)}")
            if verbose:
                imprimir_matriz(M, f"\nL{c+1} <- L{c+1} / {fmt(p)}:")
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [M[r][k] - f * M[c][k] for k in range(n + 1)]
                sinal = "-" if f > 0 else "+"
                passos.append(f"L{r+1} <- L{r+1} {sinal} {fmt(abs(f))}*L{c+1}")
                if verbose:
                    imprimir_matriz(M, f"\nL{r+1} <- L{r+1} {sinal} {fmt(abs(f))}*L{c+1}:")
    x = [M[i][n] for i in range(n)]
    return x, passos


def mat_vec(M, v):
    return [sum(Fraction(M[i][j]) * v[j] for j in range(len(v))) for i in range(len(M))]


# ---------------------------------------------------------------------------
# ETAPA 1
# ---------------------------------------------------------------------------
def etapa1(verbose=True):
    if verbose:
        secao("ETAPA 1 - Diagnóstico e modelagem da situação atual")
        print("\n(a) Variáveis: N, S, L, O = nº de ônibus alocados em Norte, Sul, Leste e Oeste")
        print("    (inteiros, não negativos).")
        print("\n(b) Modelo (Relatório I):")
        print("    N +  S +  L +  O = 72")
        print("    N + 2S      +  O = 76")
        print("         S + 2L      = 52")
        print("    N      +  L + 2O = 76")
        print("\n(c) Forma matricial A·x = b:")
        imprimir_matriz(A, "A =", larg=3)
        print("x = [N, S, L, O]^T")
        print(f"b = {b}^T")

    detA = determinante(A)
    Ab = [A[i] + [b[i]] for i in range(4)]
    rA, rAb = posto(A), posto(Ab)
    if verbose:
        print("\n(d) Existência e unicidade da solução:")
        print(f"    det(A) = {fmt(detA)}  (cálculo exato)   | numpy: {np.linalg.det(np.array(A, float)):.4f}")
        print(f"    posto(A) = {rA}, posto([A|b]) = {rAb}, nº de incógnitas = 4")
        print("    Pelo Teorema de Rouché-Capelli: posto(A) = posto([A|b]) = 4 = nº de incógnitas,")
        print("    então o sistema é possível e determinado (SPD): existe UMA única distribuição.")
        print("    Equivalente: det(A) != 0, logo A é invertível e x = A^-1 · b.")
        print("\n(e) Resolução por eliminação de Gauss-Jordan:\n")

    x, passos = gauss_jordan(A, b, verbose=verbose)

    # Conferência com numpy e com a inversa
    x_np = np.linalg.solve(np.array(A, float), np.array(b, float))
    A_inv = np.linalg.inv(np.array(A, float))
    if verbose:
        print("\nSolução (Gauss-Jordan, frações exatas):")
        for nome, v in zip(REGIOES, x):
            print(f"    {nome:6s} = {fmt(v)} ônibus")
        print(f"\nConferência numpy.linalg.solve: {np.round(x_np, 6)}")
        print(f"Conferência x = A^-1·b:         {np.round(A_inv @ np.array(b, float), 6)}")
        print("\n(f) Verificação de todos os registros operacionais:")
    ok = True
    Ax = mat_vec(A, x)
    for desc, lhs, rhs in zip(DESCRICAO_EQUACOES, Ax, b):
        bate = lhs == rhs
        ok &= bate
        if verbose:
            print(f"    {desc:26s}: {fmt(lhs):>4} = {rhs:<4} -> {'OK' if bate else 'FALHA'}")
    inteiros = all(v.denominator == 1 and v >= 0 for v in x)
    if verbose:
        print(f"    Valores inteiros e não negativos: {'OK' if inteiros else 'FALHA'}")
    assert ok and inteiros, "A solução não satisfaz os registros."
    return [int(v) for v in x], A_inv, passos


# ---------------------------------------------------------------------------
# ETAPA 2
# ---------------------------------------------------------------------------
def demanda_onibus():
    """Ônibus necessários = teto(passageiros / capacidade)."""
    return [ceil(p / c) for p, c in zip(PASSAGEIROS, CAPACIDADE)]


def plano_transferencias(atual, necessario):
    """
    Encontra o plano de transferências de custo mínimo (problema de transporte).
    - Só regiões com excedente cedem ônibus; só regiões com déficit recebem.
    - Custo = soma(qtd * custo_deslocamento) -> prefere regiões vizinhas.
    - Como a frota total = demanda total, todo déficit é coberto.
    Busca exaustiva (o problema é pequeno), garantindo o ótimo exato.
    """
    saldo = [a - n for a, n in zip(atual, necessario)]
    doadoras = [i for i, s in enumerate(saldo) if s > 0]
    receptoras = [i for i, s in enumerate(saldo) if s < 0]
    pares = [(o, d) for o in doadoras for d in receptoras]
    limites = [range(0, min(saldo[o], -saldo[d]) + 1) for o, d in pares]
    melhor, melhor_custo = None, None
    for qtds in product(*limites):
        enviado = {o: 0 for o in doadoras}
        recebido = {d: 0 for d in receptoras}
        for (o, d), q in zip(pares, qtds):
            enviado[o] += q
            recebido[d] += q
        if any(enviado[o] > saldo[o] for o in doadoras):
            continue
        if any(recebido[d] != -saldo[d] for d in receptoras):
            continue
        custo = sum(q * custo_deslocamento(SIGLAS[o], SIGLAS[d]) for (o, d), q in zip(pares, qtds))
        if melhor_custo is None or custo < melhor_custo:
            melhor, melhor_custo = qtds, custo
    F = [[0] * 4 for _ in range(4)]  # F[destino][origem]
    for (o, d), q in zip(pares, melhor):
        F[d][o] = q
    return F, melhor_custo


def matriz_transformacao(atual, F):
    """
    M[i][j] (i != j) = fração da frota da região j enviada para a região i
    M[j][j]          = fração da frota da região j que permanece em j
    Cada coluna soma 1 (matriz estocástica por colunas).
    """
    M = [[Fraction(0)] * 4 for _ in range(4)]
    for j in range(4):
        enviados = sum(F[i][j] for i in range(4) if i != j)
        for i in range(4):
            if i != j:
                M[i][j] = Fraction(F[i][j], atual[j])
        M[j][j] = 1 - Fraction(enviados, atual[j])
    return M


def aplicar_politica(M, x):
    """T(x) = M·x com arredondamento pelo maior resto (preserva o total e a integralidade)
    para vetores x diferentes da frota de referência."""
    y = mat_vec(M, [Fraction(v) for v in x])
    base = [int(v) for v in y]  # floor (valores >= 0)
    falta = sum(x) - sum(base)
    restos = sorted(range(4), key=lambda i: y[i] - base[i], reverse=True)
    for i in restos[:falta]:
        base[i] += 1
    return y, base


def etapa2(atual, verbose=True):
    necessario = demanda_onibus()
    saldo = [a - n for a, n in zip(atual, necessario)]
    if verbose:
        secao("ETAPA 2 - Política de redistribuição")
        print("\n(a) Ônibus necessários = teto(passageiros / capacidade):")
        print(f"    {'Região':7s} {'Pass.':>6s} {'Cap.':>5s} {'Necess.':>8s}")
        for r, p, c, n in zip(REGIOES, PASSAGEIROS, CAPACIDADE, necessario):
            print(f"    {r:7s} {p:6d} {c:5d} {n:8d}")
        print(f"    Total necessário = {sum(necessario)} = frota disponível ({sum(atual)})")
        print("\n(b) Desequilíbrios (atual - necessário) e taxa de ocupação no pico:")
        print(f"    {'Região':7s} {'Atual':>6s} {'Necess.':>8s} {'Saldo':>6s} {'Ocupação':>9s}")
        for r, a, n, s, p, c in zip(REGIOES, atual, necessario, saldo, PASSAGEIROS, CAPACIDADE):
            ocup = p / (a * c) * 100
            situacao = "déficit" if s < 0 else ("excedente" if s > 0 else "equilibrada")
            print(f"    {r:7s} {a:6d} {n:8d} {s:+6d} {pct(ocup):>8s}%  {situacao}")
        falta_vagas = sum(max(0, p - a * c) for p, a, c in zip(PASSAGEIROS, atual, CAPACIDADE))
        print(f"    Passageiros sem vaga na distribuição atual: {falta_vagas}")

    F, custo = plano_transferencias(atual, necessario)
    if verbose:
        print("\n(c) Política proposta - transferências de custo mínimo:")
        print("    Critérios: (1) cobrir 100% da demanda de cada região; (2) só quem tem excedente cede;")
        print("    (3) mover o menor número possível de ônibus; (4) preferir regiões vizinhas")
        print("    (custo 1 vizinha, 2 oposta). Ótimo encontrado por busca exaustiva.")
        for j in range(4):
            for i in range(4):
                if F[i][j]:
                    tipo = "vizinha" if custo_deslocamento(SIGLAS[j], SIGLAS[i]) == 1 else "oposta"
                    print(f"    {REGIOES[j]:6s} -> {REGIOES[i]:6s}: {F[i][j]} ônibus ({tipo})")
        print(f"    Ônibus deslocados: {sum(map(sum, F))} | custo total ponderado: {custo}")

    M = matriz_transformacao(atual, F)
    if verbose:
        print("\n(d)/(e) Transformação linear T: R^4 -> R^4, T(x) = M·x, com")
        imprimir_matriz(M, "M =", larg=6)
        print("    Coluna j = como a frota da região j é repartida; M[i][j] = fração que vai de j para i;")
        print("    diagonal = fração que permanece na própria região.")
        print("    M em decimais:")
        for linha in M:
            print("    [" + " ".join(f"{float(v):7.4f}" for v in linha) + " ]")

    # (f) Demonstrações
    soma_colunas = [sum(M[i][j] for i in range(4)) for j in range(4)]
    nao_neg = all(M[i][j] >= 0 for i in range(4) for j in range(4))
    nova = mat_vec(M, [Fraction(v) for v in atual])
    nova_int = [int(v) for v in nova]
    if verbose:
        print("\n(f) Propriedades:")
        print(f"    Soma de cada coluna de M: {[fmt(s) for s in soma_colunas]}  =>  1^T·M = 1^T")
        print("    Logo, para qualquer x: soma(T(x)) = 1^T·(M·x) = (1^T·M)·x = 1^T·x = soma(x).")
        print(f"    Todos os coeficientes M[i][j] >= 0: {nao_neg}")
        print("    Logo, para qualquer x >= 0: T(x)_i = soma_j M[i][j]·x_j >= 0 (soma de termos >= 0).")
        print(f"    det(M) = {fmt(determinante(M))} != 0 -> T é invertível (a política pode ser desfeita).")
        print("\nResultado: T(x_atual) =", [fmt(v) for v in nova])
        print(f"    Total antes = {sum(atual)}, total depois = {fmt(sum(nova))}")
    assert all(s == 1 for s in soma_colunas) and nao_neg
    assert sum(nova) == sum(atual) and nova_int == necessario

    # Avaliação
    if verbose:
        secao("AVALIAÇÃO DOS RESULTADOS")
        print(f"    {'Região':7s} {'Antes':>6s} {'Depois':>7s} {'Ocup. antes':>12s} {'Ocup. depois':>13s}")
        for r, a, d, p, c in zip(REGIOES, atual, nova_int, PASSAGEIROS, CAPACIDADE):
            print(f"    {r:7s} {a:6d} {d:7d} {pct(p/(a*c)*100):>11s}% {pct(p/(d*c)*100):>12s}%")
        sem_vaga_antes = sum(max(0, p - a * c) for p, a, c in zip(PASSAGEIROS, atual, CAPACIDADE))
        sem_vaga_depois = sum(max(0, p - d * c) for p, d, c in zip(PASSAGEIROS, nova_int, CAPACIDADE))
        ociosas_antes = sum(max(0, a * c - p) for p, a, c in zip(PASSAGEIROS, atual, CAPACIDADE))
        ociosas_depois = sum(max(0, d * c - p) for p, d, c in zip(PASSAGEIROS, nova_int, CAPACIDADE))
        print(f"\n    Passageiros sem vaga: {sem_vaga_antes} -> {sem_vaga_depois}")
        print(f"    Vagas ociosas:        {ociosas_antes} -> {ociosas_depois}")
        print(f"    Ônibus redistribuídos: {sum(map(sum, F))} de {sum(atual)} "
              f"({pct(sum(map(sum, F))/sum(atual)*100)}% da frota)")

        print("\n    Teste de robustez - aplicando a mesma política a outras frotas (arredondamento")
        print("    pelo maior resto para manter números inteiros):")
        for teste in ([10, 22, 16, 24], [18, 18, 18, 18], [0, 30, 12, 30]):
            y, y_int = aplicar_politica(M, teste)
            print(f"    x = {teste} (total {sum(teste)}) -> T(x) = "
                  f"[{', '.join(f'{float(v):.2f}' for v in y)}] -> inteiro {y_int} (total {sum(y_int)})")
            assert sum(y_int) == sum(teste) and min(y_int) >= 0
    return necessario, F, M, nova_int


def main():
    atual, _, _ = etapa1()
    etapa2(atual)
    print("\nTodas as verificações passaram.")


if __name__ == "__main__":
    main()
