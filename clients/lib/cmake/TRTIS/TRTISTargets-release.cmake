#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "TRTIS::request" for configuration "Release"
set_property(TARGET TRTIS::request APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(TRTIS::request PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/librequest.so"
  IMPORTED_SONAME_RELEASE "librequest.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS TRTIS::request )
list(APPEND _IMPORT_CHECK_FILES_FOR_TRTIS::request "${_IMPORT_PREFIX}/lib/librequest.so" )

# Import target "TRTIS::request_static" for configuration "Release"
set_property(TARGET TRTIS::request_static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(TRTIS::request_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/librequest_static.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS TRTIS::request_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_TRTIS::request_static "${_IMPORT_PREFIX}/lib/librequest_static.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
