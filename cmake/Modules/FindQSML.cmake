# Find the QSML libraries
#
# The following variables are optionally searched for defaults
#  QSML_ROOT:            Base directory where all QSML components are found
#
# The following are set after configuration is done:
#  QSML_FOUND
#  QSML_INCLUDE_DIRS
#  QSML_LIBRARIES
#  QSML_LIBRARYRARY_DIRS

find_path(QSML_CBLAS_INCLUDE_DIR NAMES qblas_cblas.h  PATHS "$ENV{QSML_ROOT}/include")
find_library(QSML_BLAS_LIBRARY   NAMES QSML PATHS "$ENV{QSML_ROOT}/lib")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(QSML DEFAULT_MSG QSML_CBLAS_INCLUDE_DIR QSML_BLAS_LIBRARY)

if(QSML_FOUND)
  set(QSML_INCLUDE_DIR ${QSML_CBLAS_INCLUDE_DIR})
  set(QSML_LIBRARIES ${QSML_BLAS_LIBRARY})
  mark_as_advanced(${QSML_CBLAS_INCLUDE_DIR} ${QSML_BLAS_LIBRARY})

  message(STATUS "Found QSML (include: ${QSML_CBLAS_INCLUDE_DIR}, library: ${QSML_BLAS_LIBRARY})")
endif(QSML_FOUND)

