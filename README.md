# Phase-Field Encoding

Phase field encoding is a biologically-inspired method of converting input data (like images) into temporal spike patterns, similar to how biological neurons process information.

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

---

# Ideas

- **Layerwise z tracking**:
- [ ] Save encoded_img, x1, x2, logits at each stage of the model
- [ ] Feed those into generate_phase_plot() to animate how samples evolve across the MLP

- **Trainable ω and κ**:
- [ ] Make `omega_active` and/or `kappa` learnable tensors (e.g. per-pixel, per-layer)
- [ ] Backprop through the encoding step → adapt phase timing itself for better discriminative power

- **Contrastive or metric learning**:
- [ ] Use cosine distance between z representations to train in embedding space
- Phase geometry lends itself naturally to angular margins or triplet losses

- **Layered propagation in phase domain**:
- [ ] Insert trainable complex-valued transforms between phase encoding and classification
- [ ] Build a true stackable phase computing module, not just MLP on top

Next Steps:
- [ ] Track and export intermediate layer states (z) for visualization
- [ ] Implement a learnable omega_active or spatial kappa
- [ ] Try contrastive learning on phase geometry