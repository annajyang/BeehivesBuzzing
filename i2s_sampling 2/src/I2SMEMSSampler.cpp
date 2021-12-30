#include <Arduino.h>
#include "driver/i2s.h"
#include "soc/i2s_reg.h"

#include "I2SMEMSSampler.h"

I2SMEMSSampler::I2SMEMSSampler(i2s_pin_config_t &i2sPins) {
    m_i2sPins = i2sPins;
}

void I2SMEMSSampler::configureI2S() {
    i2s_set_pin(getI2SPort(), &m_i2sPins);
}

void I2SMEMSSampler::processI2SData(uint8_t *i2sData, size_t bytesRead) {
    int32_t *samples = (int32_t *)i2sData;
    for (int i = 0; i < bytesRead / 4; i++)     {
        // you may need to vary the >> 11 to fit your volume - ideally we'd have some kind of AGC here
        addSample(samples[i] >> 11);
    }
}
