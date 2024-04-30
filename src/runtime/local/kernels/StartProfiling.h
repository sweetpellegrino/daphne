/*
 * Copyright 2023 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <iostream>
#include <runtime/local/context/DaphneContext.h>

#include <papi.h>

// ****************************************************************************
// Convenience function
// ****************************************************************************

void startProfiling(DCTX(ctx)) {
    std::cout << "START_PROFILING" << std::endl;
    int retval;
    retval = PAPI_hl_region_begin("computation");
    if ( retval != PAPI_OK )
        std::cerr << "PAPI error " << retval << ": " << PAPI_strerror(retval) << std::endl;
}