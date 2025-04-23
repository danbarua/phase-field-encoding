# Ideas

- [ ] Port to TensorFlow (for TPU support)
- [ ] Visualisations
    - [ ] TensorBoard?

---

# Phase Encoding - The Maths

Here's a LaTeX representation of the mathematical function implemented in `encode_image()`:

## Input:

- $I$: Flattened image (a vector of pixel values)
- $\omega_{active}$: Active frequency parameter
- $x$: Spatial layout parameter (a vector)

### Parameters:

- $\theta_{thresh}$: Threshold phase (set to 0.0 in the code). When a Neuron's phase has rotated through $2\pi$ to $0$
  we consider the Nueron has generated a Spike.
- $\omega_{ref}$: Reference frequency
- $n$ A constant (set to 4.0 in the code)
- $\kappa$: A constant (set to $2\pi$ in the code)

## Calculations:

#### Initial Phase: $$\theta_{init} = I \cdot 2\pi$$

#### Phase Difference: $$\Delta\theta = (\theta_{thresh} - \theta_{init} + 2\pi) \pmod{2\pi}$$

#### Spike Time: $$t_{spike} = \frac{\Delta\theta}{\omega_{active}}$$

#### Reference Phase: $$\theta_{ref} = \left(\frac{\omega_{ref} \cdot t_{spike} + \kappa \cdot x}{n}\right) \pmod{2\pi}$$

#### Final Phase Difference: $$\phi = (\theta_{thresh} - \theta_{ref} + 2\pi) \pmod{2\pi}$$

## Output:

- Encoded image: $[\cos(\phi), \sin(\phi)]$ (a concatenated vector of cosine and sine of the final phase difference)

### Formal Definition:

$$ \text{encode_image}(I, \omega_{active}, x) = [\cos(\phi), \sin(\phi)] $$

where

$$ \phi = (\theta_{thresh} - \theta_{ref} + 2\pi) \pmod{2\pi} $$

and $\theta_{ref}$ is calculated as described in the steps above.

This definition encapsulates the mathematical operations performed by the `encode_image()` function, providing a concise
and formal representation of the phase encoding process.

---

# References/Inspiration

This simple Phase-encoder is inspired by the profound work of **Zoltan Nadasdy**.

- **Nadasdy, Z.** (2009). *Information encoding and reconstruction from the phase of action potentials*. Frontiers in
  Systems
  Neuroscience. [https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/neuro.06.006.2009/full](https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/neuro.06.006.2009/full)
- **Nadasdy, Z.** (2010). *Binding by asynchrony: the neuronal phase code*. Frontiers in
  Neuroscience. [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2010.00051/full](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2010.00051/full)
- **Nadasdy, Z., et al.** (2022). *Phase coding of spatial representations in the human entorhinal cortex*. Science
  Advances. [https://www.science.org/doi/full/10.1126/sciadv.abm6081](https://www.science.org/doi/full/10.1126/sciadv.abm6081)