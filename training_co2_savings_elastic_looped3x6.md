# Estimated training CO2e savings of Elastic Looped3x6

## Result

Under the assumption that all runs use the same hardware, utilization, data
center PUE, and electricity mix, operational training CO2e is proportional to
training FLOPs. Training one **Elastic Looped3x6 (ElTab)** model instead of the
complete suite of six fixed-depth models (**Looped3x1 through Looped3x6**)
therefore reduces estimated operational training CO2e by **57.33%**.

Equivalently, ElTab uses **42.67%** of the compute and estimated operational
CO2e of the six fixed-depth training runs. The fixed-depth suite requires
**2.344 times** as much compute and estimated operational CO2e as ElTab.

| Training scenario | Total training FLOPs | Relative CO2e | Reduction from fixed-depth suite |
|---|---:|---:|---:|
| Looped3x1 + Looped3x2 + ... + Looped3x6 | 7.744 x 10^18 | 100.00% | 0.00% |
| Elastic Looped3x6 (ElTab) | 3.304 x 10^18 | 42.67% | **57.33%** |

The avoided computation is:

```text
7.744 x 10^18 - 3.304 x 10^18 = 4.440 x 10^18 FLOPs
```

The relative reduction is:

```text
(7.744 x 10^18 - 3.304 x 10^18) / (7.744 x 10^18)
    = 0.573347 = 57.33%
```

For example, if training the six fixed-depth models emits 100 kg CO2e under a
given hardware and electricity setup, training ElTab under the same setup is
estimated to emit 42.67 kg CO2e and avoid 57.33 kg CO2e.

## Comparison with each individual fixed-depth run

ElTab is more expensive than any *single* fixed-depth model. Its CO2e advantage
only applies when it replaces the complete collection of six separately
trained fixed-depth models.

| Individual model | Total training FLOPs | Individual model CO2e as a share of ElTab | ElTab CO2e change vs individual model |
|---|---:|---:|---:|
| Looped3x1 | 3.874 x 10^17 | 11.73% | 752.87% more |
| Looped3x2 | 7.487 x 10^17 | 22.66% | 341.30% more |
| Looped3x3 | 1.110 x 10^18 | 33.60% | 197.66% more |
| Looped3x4 | 1.471 x 10^18 | 44.52% | 124.61% more |
| Looped3x5 | 1.833 x 10^18 | 55.48% | 80.25% more |
| Looped3x6 | 2.194 x 10^18 | 66.40% | 50.59% more |
| Elastic Looped3x6 (ElTab) | 3.304 x 10^18 | 100.00% | baseline |

## Scope and conversion to absolute emissions

FLOPs alone support a relative CO2e estimate, not an absolute value in kg.
Absolute operational emissions additionally require the measured or estimated
energy per FLOP, data-center PUE, and grid carbon intensity:

```text
CO2e [kg] = FLOPs x energy_per_FLOP [J/FLOP] x PUE
            x carbon_intensity [kg CO2e/kWh] / 3.6 x 10^6
```

The 57.33% reduction assumes those factors are equal across all compared runs.
It excludes embodied hardware emissions and checkpoint-storage emissions.

## Paper-ready statement

> Assuming identical hardware and electricity conditions, operational training
> emissions scale with compute. ElTab requires 3.304 x 10^18 FLOPs, compared
> with 7.744 x 10^18 FLOPs for training the six fixed-depth Looped3x1--3x6
> models separately. It therefore reduces estimated operational training CO2e
> by 57.33% (4.440 x 10^18 fewer FLOPs), using 42.67% of the compute of the
> fixed-depth suite.

Source values: the supplied training-computation table (200 epochs and 4,672
microbatches per epoch).
