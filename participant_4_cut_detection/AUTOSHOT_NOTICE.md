# AutoShot architecture notice

`autoshot_architecture.py` is a compatibility implementation for the AutoShot
network published by Wentao Zhu and contributors:

- Project: `wentaozhu/AutoShot`
- Paper: *AutoShot: A Short Video Dataset and State-of-the-Art Shot Boundary Detection*
- Upstream license: MIT

The implementation is bundled locally so the worker does not download and
execute Python source during container startup. The supplied checkpoint remains
subject to the terms under which the project team received it.
