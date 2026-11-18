# Citizen Assist AI Mobilidade — Modelagem Matemática e Computacional

**FIAP · Ciência da Computação 1CCR · Global Solutions 2 (2026)**

Modelo da frota de ônibus do projeto **Citizen Assist AI Mobilidade** (Global Solution *Mobilidade Urbana Inteligente: Modelagem Matemática para a Gestão do Transporte Público*). O modelo faz três coisas:

1. reconstrói a distribuição atual de 72 ônibus entre as regiões Norte, Sul, Leste e Oeste com um **sistema linear** (Etapa 1);
2. estima a necessidade de ônibus de cada região no horário de pico;
3. propõe uma **política de redistribuição** representada por uma **transformação linear** que preserva o total da frota e nunca gera alocações negativas (Etapa 2).

| Integrante | RM |
|---|---|
| Gabriel Fagundes | 569074 |
| Glauco Kelly | 572840 |
| Thiago Renatino | 569073 |

## Estrutura do repositório

```
├── frota.py             # modelo completo: Etapa 1 e Etapa 2, com todas as verificações
├── requirements.txt     # dependência: numpy
├── saida_execucao.txt   # saída completa de python frota.py
└── docs/
    └── Relatorio_Modelagem_Matematica_Frota.pdf
```

## Como executar

Requisito: **Python 3.10 ou superior**.

```bash
git clone https://github.com/FIAP-GS-02-2026/citizen-assist-ai-mmc.git
cd citizen-assist-ai-mmc

pip install -r requirements.txt
python frota.py
```

A execução termina com **"Todas as verificações passaram."**. No Windows, se os acentos aparecerem errados no terminal, execute antes `set PYTHONIOENCODING=utf-8` (Prompt de Comando) ou `$env:PYTHONIOENCODING="utf-8"` (PowerShell).

## Etapa 1 — Diagnóstico da situação atual

Variáveis: x = (N, S, L, O) = número de ônibus em cada região.

```
N +  S +  L +  O = 72     (frota total)
N + 2S      +  O = 76     (Norte + Oeste + dobro do Sul)
     S + 2L      = 52     (Sul + dobro do Leste)
N      +  L + 2O = 76     (Norte + Leste + dobro do Oeste)
```

- det(A) = 3 ≠ 0 e posto(A) = posto([A|b]) = 4. Pelo Teorema de Rouché-Capelli, o sistema é **possível e determinado**: existe uma única distribuição.
- Resolvido por **Gauss-Jordan com frações exatas** (conferido com `numpy.linalg.solve` e com A⁻¹·b): **N = 12, S = 20, L = 16, O = 24**.
- Os quatro registros operacionais conferem.

## Etapa 2 — Política de redistribuição

- Necessidade = teto(passageiros / 40) = **(14, 19, 18, 21)**, total 72.
- Desequilíbrios: falta 2 no Norte e 2 no Leste; sobra 1 no Sul e 3 no Oeste. 160 passageiros ficam sem vaga no pico.
- Política (mínimo de ônibus movidos, preferindo regiões vizinhas): **Oeste → Norte 2, Sul → Leste 1, Oeste → Leste 1**.

```
          de N   de S    de L   de O
M = [     1      0       0      1/12 ]  para N
    [     0      19/20   0      0    ]  para S
    [     0      1/20    1      1/24 ]  para L
    [     0      0       0      7/8  ]  para O
```

- Cada coluna soma 1, então 1ᵀM = 1ᵀ e a transformação **preserva o total** de ônibus.
- Todas as entradas são ≥ 0, então a transformação **nunca gera alocações negativas**.
- T(12, 20, 16, 24) = **(14, 19, 18, 21)**: passageiros sem vaga no pico caem de 160 para 0, com apenas 4 ônibus deslocados (5,6% da frota).

O relatório completo, com as demonstrações, os gráficos e as limitações, está em [`docs/Relatorio_Modelagem_Matematica_Frota.pdf`](docs/Relatorio_Modelagem_Matematica_Frota.pdf).

Os dados são hipotéticos e representam uma simplificação da operação real de transporte público.
