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

#include <memory>
#include <stdexcept>

#include "../../AI/MMAI/schema/schema.h"
#include "TorchModel.h"

namespace MMAI {

#ifdef ENABLE_LIBTORCH
    TorchModel::TorchModel(std::string path, bool verbose)
    : verbose(verbose)
    {
        c10::InferenceMode guard;
        model = torch::jit::load(path);
        model.eval();

        version = model.get_method("get_version")({}).toInt();

        switch(version) {
            break; case 1:
                sizeOneHex = MMAI::Schema::V1::BATTLEFIELD_STATE_SIZE_ONE_HEX;
                nactions = MMAI::Schema::V1::N_ACTIONS;
            break; case 2:
                sizeOneHex = MMAI::Schema::V2::BATTLEFIELD_STATE_SIZE_ONE_HEX;
                nactions = MMAI::Schema::V1::N_ACTIONS;
            break; case 3:
                sizeOneHex = MMAI::Schema::V3::BATTLEFIELD_STATE_SIZE_ONE_HEX;
                nactions = MMAI::Schema::V3::N_ACTIONS;
            break; default:
                throw std::runtime_error("Unknown MMAI version: " + std::to_string(version));
        }

        auto out_features = model.attr("actor").toModule().attr("out_features").toInt();

        switch(out_features) {
        break; case 2311: actionOffset = 1;
        break; case 2312: actionOffset = 0;
        break; default:
            throw std::runtime_error("Expected 2311 or 2312 out_features for actor, got: " + std::to_string(out_features));
        }
    }

    std::string TorchModel::getName() {
        return "MMAI_MODEL";
    };

    int TorchModel::getVersion() {
        return version;
    };

    int TorchModel::getAction(const MMAI::Schema::IState * s) {
        c10::InferenceMode guard;
        auto any = s->getSupplementaryData();
        auto ended = false;

        switch(version) {
            break; case 1:
                   case 2:
                ended = std::any_cast<const MMAI::Schema::V1::ISupplementaryData*>(any)->getIsBattleEnded();
            break; case 3:
                ended = std::any_cast<const MMAI::Schema::V3::ISupplementaryData*>(any)->getIsBattleEnded();
            break; default:
                throw std::runtime_error("Unknown MMAI version: " + std::to_string(version));
        }

        if (ended)
            return MMAI::Schema::ACTION_RESET;

        auto &src = s->getBattlefieldState();
        auto dst = MMAI::Schema::BattlefieldState{};
        dst.resize(src.size());
        std::copy(src.begin(), src.end(), dst.begin());

        auto obs = version < 3
            ? torch::from_blob(dst.data(), {11, 15, sizeOneHex}, torch::kFloat)
            : torch::from_blob(dst.data(), {static_cast<long long>(dst.size())}, torch::kFloat);

        // yields no performance benefit over (safer) copy approach:
        // auto obs = torch::from_blob(const_cast<float*>(s->getBattlefieldState().data()), {11, 15, sizeOneHex}, torch::kFloat);

        auto intmask = std::vector<int>{};
        intmask.reserve(nactions);
        auto &boolmask = s->getActionMask();

        for (auto it = boolmask.begin() + actionOffset; it != boolmask.end(); ++it)
            intmask.push_back(static_cast<int>(*it));

        auto mask = torch::from_blob(intmask.data(), {static_cast<long>(intmask.size())}, torch::kInt).to(torch::kBool);

        // auto mask_accessor = mask.accessor<bool,1>();
        // for (int i = 0; i < mask_accessor.size(0); ++i)
        //     printf("mask[%d]=%d\n", i, mask_accessor[i]);

        auto method = model.get_method("predict");
        auto inputs = std::vector<torch::IValue>{obs, mask};
        auto res = method(inputs).toInt() + actionOffset;

        if (verbose) {
            printf("AI action prediction: %d\n", int(res));

            // Also esitmate value
            auto vmethod = model.get_method("get_value");
            auto vinputs = std::vector<torch::IValue>{obs};
            auto vres = vmethod(vinputs).toDouble();
            printf("AI value estimation: %f\n", vres);
        }

        return MMAI::Schema::Action(res);
    };

    double TorchModel::getValue(const MMAI::Schema::IState * s) {
        c10::InferenceMode guard;
        auto &src = s->getBattlefieldState();
        auto dst = MMAI::Schema::BattlefieldState{};
        dst.reserve(dst.size());
        std::copy(src.begin(), src.end(), dst.begin());
        auto obs = torch::from_blob(dst.data(), {11, 15, sizeOneHex}, torch::kFloat);

        auto method = model.get_method("get_value");
        auto inputs = std::vector<torch::IValue>{obs};
        auto res = method(inputs).toDouble();
        return res;
    }
#else
    TorchModel::TorchModel(std::string path, bool verbose) {
        throw std::runtime_error(
            "This binary was compiled without the ENABLE_LIBTORCH flag"
            " and cannot load \"MMAI_MODEL\" files."
        );
    }

    std::string TorchModel::getName() { return ""; };
    int TorchModel::getVersion() { return 0; }
    int TorchModel::getAction(const MMAI::Schema::IState * s) { return 0; }
    double TorchModel::getValue(const MMAI::Schema::IState * s) { return 0; }
#endif // ENABLE_LIBTORCH
}
