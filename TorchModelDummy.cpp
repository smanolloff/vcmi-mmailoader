#include <memory>
#include <stdexcept>

#include "../AI/MMAI/schema/schema.h"
#include "TorchModel.h"

/*
 * Dummy implementation of TorchModel:
 * Used if ENABLE_LIBTORCH is not defined (see notes in CMakeLists.txt)
 */

namespace MMAI {
    class TorchModel::TorchJitImpl {};

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
}
