# Copyright 2010-2024 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#[=======================================================================[.rst:
FindMosek
--------

This module determines the Mosek library of the system.

IMPORTED Targets
^^^^^^^^^^^^^^^^

This module defines :prop_tgt:`IMPORTED` target ``mosek::mosek``, if
Mosek has been found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

::

MOSEK_FOUND          - True if Mosek found.

Hints
^^^^^

A user may set ``MOSEK_PLATFORM_DIR`` to a Mosek installation platform
directoru to tell this module where to look.
#]=======================================================================]
set(MOSEK_FOUND FALSE)
message(STATUS "Locating MOSEK")

if(CMAKE_C_COMPILER_LOADED)
  include (CheckIncludeFile)
  include (CheckCSourceCompiles)
elseif(CMAKE_CXX_COMPILER_LOADED)
  include (CheckIncludeFileCXX)
  include (CheckCXXSourceCompiles)
else()
  message(FATAL_ERROR "FindMosek only works if either C or CXX language is enabled")
endif()

if(APPLE) 
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)")
        SET(MOSEK_PLATFORM_NAME "osxaarch64")
    elseif()
        message(FATAL_ERROR "Mosek not supported for ${CMAKE_SYSTEM} / ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
elseif(UNIX)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)")
        SET(MOSEK_PLATFORM_NAME "linuxaarch64")
    else()
        SET(MOSEK_PLATFORM_NAME "linux64x86")
    endif()
elseif(MSVC)
    SET(MOSEK_PLATFORM_NAME "win64x86")
else()
    message(FATAL_ERROR "Mosek not supported for ${CMAKE_SYSTEM}")
endif()

function(FindMosekPlatformInPath RESULT PATH)
    if(EXISTS "${PATH}/mosek")
        SET(dirlist "")
        if(EXISTS "$ENV{HOME}/mosek")
            FILE(GLOB entries LIST_DIRECTORIES true "$ENV{HOME}/mosek/*")
            FOREACH(f ${entries})
                if(IS_DIRECTORY "${f}")
                    get_filename_component(bn "${f}" NAME)
                    if("${bn}" MATCHES "^[0-9]+[.][0-9]+$") 
                        if (${bn} GREATER_EQUAL "10.0")
                            LIST(APPEND dirlist "${bn}")
                        endif()
                    endif()
                endif()
            ENDFOREACH()
            LIST(SORT dirlist COMPARE NATURAL ORDER DESCENDING)
        endif()

        LIST(LENGTH dirlist dirlistlen)
        IF(dirlistlen GREATER 0)
            LIST(GET dirlist 0 MOSEK_VERSION)
            if(MOSEK_PLATFORM_NAME)
                SET("${RESULT}" "${PATH}/mosek/${MOSEK_VERSION}/tools/platform/{MOSEK_PLATFORM_NAME}")
                return(PROPAGATE "${RESULT}")
            endif()
        endif()
    endif()
endfunction()


# Where to look for MOSEK:
# Standard procedure in Linux/OSX is to install MOSEK in the home directory, i.e.
#   $HOME/mosek/X.Y/...
# Option 1. The user can specify when running CMake where the MOSEK platform directory is located, e.g.
#     -DMOSEK_PLATFORM_DIR=$HOME/mosek/10.2/tools/platform/linux64x86/ 
#   in which case no search is performed.
# Option 2. The user can specify MOSEK_ROOT when running cmake. MOSEK_ROOT is
#   the directory where the root mosek/ tree is located.
# Option 3. Automatic search. We will then attempt to search in the default
#   locations, and if that fails, assume it is installed in a system location.
# For option 2 and 3, the newest MOSEK version will be chosen if more are available.

if(MOSEK_PLATFORM_DIR)
    # User defined platform dir directly
elseif(MOSEK_ROOT)
    FindMosekPlatformInPath("${MOSEK_ROOT}/mosek" MOSEK_PLATFORM_DIR)
endif()
if(NOT MOSEK_PLATFORM_NAME)

endif()




if(NOT MOSEK_PLATFORM_DIR)
    message(FATAL_ERROR "MOSEK_PLATFORM_DIR: not found")
else()
  message(STATUS "MOSEK_PLATFORM_DIR detected: ${MOSEK_PLATFORM_DIR}")
  set(MOSEK_PFDIR_FOUND TRUE)
endif()


if(MOSEK_PFDIR_FOUND AND NOT TARGET Mosek::Mosek)
  add_library(mosek::mosek UNKNOWN IMPORTED)
  
  find_path(MOSEKINC mosek.h HINTS ${MOSEK_PLATFORM_DIR}/h)
  find_library(LIBMOSEK mosek64 HINTS ${MOSEK_PLATFORM_DIR}/bin)

  if(LIBMOSEK) 
      set_target_properties(mosek::mosek PROPERTIES IMPORTED_LOCATION ${LIBMOSEK})
  endif()

  if (MOSEKINC)
      target_include_directories(mosek::mosek INTERFACE "${MOSEKINC}")
  endif()
elseif(NOT TARGET Mosek::Mosek)
  add_library(mosek::mosek UNKNOWN IMPORTED)
  
  find_path(MOSEKINC mosek.h)
  find_library(LIBMOSEK mosek64)

  if(LIBMOSEK) 
      set_target_properties(mosek::mosek PROPERTIES IMPORTED_LOCATION ${LIBMOSEK})
  endif()

  if (MOSEKINC)
      target_include_directories(mosek::mosek INTERFACE "${MOSEKINC}")
  endif()
endif()
