#include "encoding_factory.h"
#include "hash_grid_encoding.h"
#include "identity_encoding.h"
#include "sh_encoding.h"
#include "frequency_encoding.h"
#include "oneblob_encoding.h"

#include <stdexcept>

std::unique_ptr<Encoding> createEncoding(Device &device, const std::string &type,
                                         int inputDim, const json &params)
{
    if (type == "hashgrid")
    {
        HashGridEncoding::Config cfg;
        cfg.inputDim = inputDim;
        if (params.contains("numLevels"))          cfg.numLevels = params["numLevels"].get<int>();
        if (params.contains("featuresPerLevel"))   cfg.featuresPerLevel = params["featuresPerLevel"].get<int>();
        if (params.contains("tableSize"))          cfg.tableSize = params["tableSize"].get<int>();
        if (params.contains("coarsestResolution")) cfg.coarsestResolution = params["coarsestResolution"].get<int>();
        if (params.contains("finestResolution"))   cfg.finestResolution = params["finestResolution"].get<int>();
        return std::make_unique<HashGridEncoding>(device, cfg);
    }

    if (type == "sh")
    {
        SHEncoding::Config cfg;
        cfg.inputDim = inputDim;
        if (params.contains("degree")) cfg.degree = params["degree"].get<int>();
        return std::make_unique<SHEncoding>(device, cfg);
    }

    if (type == "frequency")
    {
        FrequencyEncoding::Config cfg;
        cfg.inputDim = inputDim;
        if (params.contains("numFreqs")) cfg.numFreqs = params["numFreqs"].get<int>();
        return std::make_unique<FrequencyEncoding>(device, cfg);
    }

    if (type == "oneblob")
    {
        OneBlobEncoding::Config cfg;
        cfg.inputDim = inputDim;
        if (params.contains("numBins")) cfg.numBins = params["numBins"].get<int>();
        if (params.contains("sigma"))   cfg.sigma = params["sigma"].get<float>();
        return std::make_unique<OneBlobEncoding>(device, cfg);
    }

    if (type == "identity")
    {
        IdentityEncoding::Config cfg;
        cfg.inputDim = inputDim;
        return std::make_unique<IdentityEncoding>(device, cfg);
    }

    throw std::runtime_error("Unknown encoding type: " + type);
}
