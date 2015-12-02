###############################################################################
# Find iSAM
#
# This sets the following variables:
# ISAM_FOUND - True if ISAM was found.
# ISAM_INCLUDE_DIRS - Directories containing the ISAM include files.
# ISAM_LIBRARIES - Libraries needed to use ISAM.

find_path(ISAM_INCLUDE_DIR isam.h
          PATHS
            /usr/include
            /usr/local/include
          PATH_SUFFIXES isam
)

find_library(ISAM_LIBRARY
             NAMES libisam.a
             PATHS
               /usr/lib
               /usr/local/lib
             PATH_SUFFIXES ${ISAM_PATH_SUFFIXES}
)

set(ISAM_INCLUDE_DIRS ${ISAM_INCLUDE_DIR})
set(ISAM_LIBRARIES ${ISAM_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(iSAM DEFAULT_MSG ISAM_LIBRARY ISAM_INCLUDE_DIR)
mark_as_advanced(ISAM_LIBRARY ISAM_INCLUDE_DIR)
