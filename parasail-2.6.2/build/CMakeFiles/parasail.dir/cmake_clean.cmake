file(REMOVE_RECURSE
  "libparasail.pdb"
  "libparasail.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang C)
  include(CMakeFiles/parasail.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
