// =============================================================================
// Copyright 2024 Simeon Manolov <s.manolloff@gmail.com>.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "TorchModel.h"
#include <stdexcept>
#ifdef ENABLE_LIBTORCH
#include <ATen/core/enum_tag.h>
#include <ATen/core/ivalue.h>
#include <c10/core/SymFloat.h>
#include <c10/core/ScalarType.h>
#include <torch/torch.h>
#include <torch/script.h>
#endif

#include "../../AI/MMAI/schema/schema.h"

#include "Loader.h"
#include <string>

namespace MMAI {
    Schema::Baggage * LoadModels(
        std::string leftModelPath,
        std::string rightModelPath,
        bool verbose
    ) {
        auto res = new Schema::Baggage();
        res->modelLeft = new TorchModel(leftModelPath, verbose);
        res->modelRight = new TorchModel(rightModelPath, verbose);
        return res;
    }
}

