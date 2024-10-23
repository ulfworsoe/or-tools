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


if(NOT MOSEK_PLATFORM_DIR)
  if(DEFINED ENV{MOSEK_BIN_DIR})
    set(MOSEK_PLATFORM_DIR ENV{MOSEK_BIN_DIR}/..)
  elseif(DEFINED ENV{HOME}) 
      SET(dirlist "")
      if(EXISTS "$ENV{HOME}/mosek")
          FILE(GLOB entries LIST_DIRECTORIES true "$ENV{HOME}/mosek/[0-9]*.[0-9]*")          
          FOREACH(f ${entries})
              if(IS_DIRECTORY "${f}")
                  get_filename_component(bn "${f}" NAME)
                  LIST(APPEND dirlist "${bn}")
              endif()
          ENDFOREACH()
          LIST(SORT dirlist COMPARE NATURAL ORDER DESCENDING)          
      endif()
      
      LIST(GET dirlist 0 MOSEK_VERSION)

      if(APPLE) 
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)")
            SET(MOSEK_PLATFORM_DIR "$ENV{HOME}/mosek/${MOSEK_VERSION}/tools/platform/osxaarch64")
        elseif()
            message(FATAL_ERROR "Mosek not supported for ${CMAKE_SYSTEM} / ${CMAKE_SYSTEM_PROCESSOR}")
        endif()
      elseif(UNIX)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)")
            SET(MOSEK_PLATFORM_DIR "$ENV{HOME}/mosek/${MOSEK_VERSION}/tools/platform/linuxaarch64")
        else()
            SET(MOSEK_PLATFORM_DIR "$ENV{HOME}/mosek/${MOSEK_VERSION}/tools/platform/linux64x86")
        endif()
      elseif(MSVC)
          SET(MOSEK_PLATFORM_DIR "$ENV{HOME}/mosek/${MOSEK_VERSION}/tools/platform/win64x86")
      else()
        message(FATAL_ERROR "Mosek not supported for ${CMAKE_SYSTEM}")
      endif()
  endif()
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
