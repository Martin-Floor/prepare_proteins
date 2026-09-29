# Relative Paths Manifesto

`prepare_proteins` setup methods should keep generated job folders portable without inventing
special launcher semantics.

Rules:

1. Paths written inside generated setup trees must be relative whenever they refer to files or
folders within the same setup tree.
2. Shell commands returned by setup methods must use plain relative paths for directories and files
created under the setup folder.
3. Returned shell commands should follow the existing library convention: a simple `cd` into the
relative job directory, run the program, then `cd` back with a simple relative path such as
`cd ../../..`.
4. Setup methods should not inject launcher-location variables or shell wrappers to reinterpret
their own relative paths. If a launcher script is generated, it should be written in the folder
from which those relative commands are expected to run.
5. Absolute paths are only acceptable for external resources that cannot be staged inside the setup
tree.

Implications:

- `setUpRosettaFlexPepDock()` should return commands like
  `cd flexpepdock_setup/output_models/model_name`, not commands based on shell variables such as a
  launcher directory prefix.
- Notebook or workflow wrappers should not patch returned job strings. They should place generated
  launcher scripts in the folder where the relative commands are meant to run.
- New setup methods should preserve the established simple relative-path style unless there is a
  documented reason not to.
