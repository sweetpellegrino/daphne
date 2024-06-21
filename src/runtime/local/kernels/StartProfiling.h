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

#ifdef USE_PAPI
#include <papi.h>
#include <stdlib.h>
#endif

// ****************************************************************************
// Convenience function
// ****************************************************************************

void startProfiling(DCTX(ctx)) {
#ifdef USE_PAPI
    //PAPI_set_debug( PAPI_VERB_ECONT );
    //putenv("PAPI_EVENTS=\"PAPI_TOT_INS,PAPI_TOT_CYC\"");
    int retval;
/*    retval = PAPI_library_init(PAPI_VER_CURRENT);
    std::cout << retval << std::endl;
    if (retval != PAPI_VER_CURRENT)
        std::cerr << "PAPI error " << retval << ": " << PAPI_strerror(retval) << std::endl;
*/
    std::cout << "START_PROFILING" << std::endl;
    retval = PAPI_hl_region_begin("fixme");
    if ( retval != PAPI_OK )
        std::cerr << "PAPI error " << retval << ": " << PAPI_strerror(retval) << std::endl;
#else
    throw std::runtime_error("daphne was built without support for PAPI");
#endif
}