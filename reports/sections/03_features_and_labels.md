# Candidate features and weak labels

For each observation a spatial index finds nearby directed segments in a common trip-local metric plane. The generator checks distance to the segment interior, retains up to eight arcs within 50 m, and uses the nearest segment as a fallback if the search finds none. Candidate exports key rows by trip, observation, and arc. Opposite directions on the same physical road remain distinct.

The nine features are point-to-segment distance, projection fraction, candidate rank, arc length, GPS displacement, GPS speed, heading error, incoming degree at the candidate start, and outgoing degree at the candidate end. The shared processor can build these features without reference routes.

The weak-label matcher first finds an unambiguous reference arc within 30 m. A competing reference arc must be more than 5 m farther away. Repeated reference arcs and matches that disagree with route order are unresolved. If the selected reference arc appears among the generated candidates, it receives label 1 and the other candidates at that observation receive 0. All labels for an unresolved or missed observation remain blank.

| Measure | Observations | Rate |
| --- | ---: | ---: |
| Training GPS observations with weak labels | 121,337 / 208,417 | 58.22% |
| Training candidate recall among independent weak reference matches | 121,337 / 121,404 | 99.94% |
| Student-test GPS observations with weak labels | 10,271 / 19,160 | 53.61% |

Candidate recall first matches against the reference independently of the candidate set, so the denominator includes the 67 resolvable training observations where the generator missed the reference arc. The coverage and recall rates apply to this weak-label method.

See the [shared feature contract](../../docs/feature_contract.md).
