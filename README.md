# ML in PL tutorial workspace

The tutorial lives in [tutorial-modules/](tutorial-modules/README.md), with one
self-contained uv project per practical session:

- [Module 1: IdeGYM sandboxes](tutorial-modules/module-1/README.md): drive sandboxes from a laptop to a cluster, work with files and datasets, build images, and take snapshots.
- [Module 2: NeMo Gym](tutorial-modules/module-2/README.md): four runnable evaluations with IDEGym sandbox verification and [timestamped run bundles](tutorial-modules/module-2/README.md#output-layout) containing logs, results, and generated summaries.

Each module owns its setup, dependencies, task inputs, and documentation. Git and
repository-wide ignore rules remain at the root.
