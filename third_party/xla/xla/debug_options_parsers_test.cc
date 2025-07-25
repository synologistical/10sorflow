/* Copyright 2017 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Test for parse_flags_from_env.cc

#include "xla/debug_options_parsers.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/debug_options_flags.h"
#include "xla/parse_flags_from_env.h"
#include "xla/service/dump.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/util/command_line_flags.h"
#include "xla/xla.pb.h"

namespace xla {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::tsl::testing::IsOkAndHolds;
using ::tsl::testing::StatusIs;
using ::xla::details::ParseRepeatedEnumModifiers;
using ::xla::details::RepeatedFlagModifier;

void SetXlaFlagsEnvVar(const std::string& value) {
  int* pargc;
  std::vector<char*>* pargv;
  ResetFlagsFromEnvForTesting("XLA_FLAGS", &pargc, &pargv);
  tsl::setenv("XLA_FLAGS", value.c_str(), true /*overwrite*/);
}

// Test that the xla_backend_extra_options flag is parsed correctly.
TEST(DebugOptionsFlags, ParseXlaBackendExtraOptions) {
  absl::flat_hash_map<std::string, std::string> test_map;
  std::string test_string = "aa=bb,cc,dd=,ee=ff=gg";
  parse_xla_backend_extra_options(&test_map, test_string);
  EXPECT_EQ(test_map.size(), 4);
  EXPECT_EQ(test_map.at("aa"), "bb");
  EXPECT_EQ(test_map.at("cc"), "");
  EXPECT_EQ(test_map.at("dd"), "");
  EXPECT_EQ(test_map.at("ee"), "ff=gg");
}

struct UppercaseStringSetterTestSpec {
  std::string user_max_isa;
  std::string expected_max_isa;
};

class UppercaseStringSetterTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<UppercaseStringSetterTestSpec> {
 public:
  UppercaseStringSetterTest()
      : flag_values_(DefaultDebugOptionsIgnoringFlags()) {
    MakeDebugOptionsFlags(&flag_objects_, &flag_values_);
  }
  static std::string Name(
      const ::testing::TestParamInfo<UppercaseStringSetterTestSpec>& info) {
    return info.param.user_max_isa;
  }
  DebugOptions flag_values() const { return flag_values_; }
  std::vector<tsl::Flag> flag_objects() { return flag_objects_; }

 private:
  DebugOptions flag_values_;
  std::vector<tsl::Flag> flag_objects_;
};

TEST_P(UppercaseStringSetterTest, XlaCpuMaxIsa) {
  UppercaseStringSetterTestSpec spec = GetParam();
  SetXlaFlagsEnvVar(absl::StrCat("--xla_cpu_max_isa=", spec.user_max_isa));
  ParseFlagsFromEnvAndDieIfUnknown("XLA_FLAGS", flag_objects());
  EXPECT_EQ(flag_values().xla_cpu_max_isa(), spec.expected_max_isa);
}

std::vector<UppercaseStringSetterTestSpec> GetUppercaseStringSetterTestCases() {
  return std::vector<UppercaseStringSetterTestSpec>({
      UppercaseStringSetterTestSpec{"sse4_2", "SSE4_2"},
      UppercaseStringSetterTestSpec{"aVx512", "AVX512"},
      UppercaseStringSetterTestSpec{"AMx_fP16", "AMX_FP16"},
  });
}

INSTANTIATE_TEST_SUITE_P(
    UppercaseStringSetterTestInstantiation, UppercaseStringSetterTest,
    ::testing::ValuesIn(GetUppercaseStringSetterTestCases()),
    UppercaseStringSetterTest::Name);

TEST(FuelTest, FuelPassCountsAreSeparate) {
  tsl::setenv("XLA_FLAGS", "--xla_fuel=ABC=1,PQR=2", /*overwrite=*/true);
  // Parse flags from the environment variable.
  int* pargc;
  std::vector<char*>* pargv;
  ResetFlagsFromEnvForTesting("XLA_FLAGS", &pargc, &pargv);
  ParseDebugOptionFlagsFromEnv(false);

  EXPECT_TRUE(ConsumeFuel("ABC"));
  EXPECT_FALSE(ConsumeFuel("ABC"));

  EXPECT_TRUE(ConsumeFuel("PQR"));
  EXPECT_TRUE(ConsumeFuel("PQR"));
  EXPECT_FALSE(ConsumeFuel("PQR"));
}

TEST(FuelTest,
     PassFuelIsSetReturnsTrueOnExplicitlyFueledPassesAndFalseOtherwise) {
  tsl::setenv("XLA_FLAGS", "--xla_fuel=MNO=1,XYZ=2", /*overwrite=*/true);
  // Parse flags from the environment variable.
  int* pargc;
  std::vector<char*>* pargv;
  ResetFlagsFromEnvForTesting("XLA_FLAGS", &pargc, &pargv);
  ParseDebugOptionFlagsFromEnv(true);
  EXPECT_FALSE(PassFuelIsSet("ABC"));
  EXPECT_TRUE(PassFuelIsSet("MNO"));
  EXPECT_FALSE(PassFuelIsSet("PQR"));
  EXPECT_TRUE(PassFuelIsSet("XYZ"));
}

std::string WriteDebugOptionsToTempFile(const DebugOptions& debug_options,
                                        std::string* contents) {
  *contents = GetNonDefaultDebugOptions(debug_options);
  tsl::Env* env = tsl::Env::Default();
  std::string fname;
  EXPECT_TRUE(env->LocalTempFilename(&fname));
  EXPECT_TRUE(
      tsl::WriteStringToFile(tsl::Env::Default(), fname, *contents).ok());
  return fname;
}

TEST(ParsingDebugOptionsTest, FailedParsing) {
  tsl::Env* env = tsl::Env::Default();
  std::string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  // The debug options file does not exist.
  EXPECT_FALSE(ParseFlagsFromDebugOptionsFile(fname));
}

TEST(ParsingDebugOptionsTest, ParsingRepeatedFields) {
  // Using the flag xla_gpu_disable_async_collectives
  // Sanity checks: default value: empty
  DebugOptions debug_options = DefaultDebugOptionsIgnoringFlags();
  EXPECT_TRUE(debug_options.xla_gpu_disable_async_collectives().empty());
  debug_options.add_xla_gpu_disable_async_collectives(DebugOptions::ALLGATHER);
  debug_options.add_xla_gpu_disable_async_collectives(
      DebugOptions::REDUCESCATTER);

  ResetFlagValues();
  std::string contents;
  ASSERT_TRUE(ParseFlagsFromDebugOptionsFile(
      WriteDebugOptionsToTempFile(debug_options, &contents)));
  DebugOptions parsed_debug_options = GetDebugOptionsFromFlags();
  EXPECT_TRUE(absl::StrContains(
      contents, "xla_gpu_disable_async_collectives: ALLGATHER"));
  EXPECT_TRUE(absl::StrContains(
      contents, "xla_gpu_disable_async_collectives: REDUCESCATTER"));
  EXPECT_EQ(parsed_debug_options.xla_gpu_disable_async_collectives_size(), 2);
  EXPECT_EQ(parsed_debug_options.xla_gpu_disable_async_collectives(0),
            DebugOptions::ALLGATHER);
  EXPECT_EQ(parsed_debug_options.xla_gpu_disable_async_collectives(1),
            DebugOptions::REDUCESCATTER);

  // We are not resetting the flags here, because we want to ensure that
  // [ALLGATHER, REDUCESCATTER] gets overwritted to [ALLTOALL] and not simply
  // appended.
  debug_options.clear_xla_gpu_disable_async_collectives();
  debug_options.add_xla_gpu_disable_async_collectives(DebugOptions::ALLTOALL);
  ASSERT_TRUE(ParseFlagsFromDebugOptionsFile(
      WriteDebugOptionsToTempFile(debug_options, &contents)));
  parsed_debug_options = GetDebugOptionsFromFlags();
  EXPECT_TRUE(absl::StrContains(contents,
                                "xla_gpu_disable_async_collectives: ALLTOALL"));
  EXPECT_FALSE(absl::StrContains(
      contents, "xla_gpu_disable_async_collectives: ALLGATHER"));
  EXPECT_FALSE(absl::StrContains(
      contents, "xla_gpu_disable_async_collectives: REDUCESCATTER"));
  EXPECT_EQ(parsed_debug_options.xla_gpu_disable_async_collectives_size(), 1);
  EXPECT_EQ(parsed_debug_options.xla_gpu_disable_async_collectives(0),
            DebugOptions::ALLTOALL);
}

TEST(ParsingDebugOptionsTest, ParseFromDebugOptionsFile) {
  // Sanity checks: The test needs to use two flags that have false and true
  // default values.
  // Default value of xla_hlo_pass_fix_detect_cycles is false.
  // Default value of xla_dump_hlo_as_long_text is true.
  DebugOptions debug_options = DefaultDebugOptionsIgnoringFlags();
  EXPECT_TRUE(debug_options.xla_dump_hlo_as_long_text());
  EXPECT_FALSE(debug_options.xla_hlo_pass_fix_detect_cycles());

  // default value (should not be in debug_options file).
  debug_options.set_xla_dump_hlo_as_long_text(true);
  // non-default value (should be in debug_options file).
  debug_options.set_xla_hlo_pass_fix_detect_cycles(true);
  std::string contents;
  auto temp_file = WriteDebugOptionsToTempFile(debug_options, &contents);
  ResetFlagValues();
  ASSERT_TRUE(ParseFlagsFromDebugOptionsFile(temp_file));
  auto parsed_debug_options = GetDebugOptionsFromFlags();
  EXPECT_TRUE(absl::StrContains(contents, "xla_hlo_pass_fix_detect_cycles"));
  EXPECT_FALSE(absl::StrContains(contents, "xla_dump_hlo_as_long_text"));
  EXPECT_TRUE(parsed_debug_options.xla_hlo_pass_fix_detect_cycles());
  EXPECT_TRUE(parsed_debug_options.xla_dump_hlo_as_long_text());

  // non-default value (should be in debug_options file).
  debug_options.set_xla_dump_hlo_as_long_text(false);
  // default value (should not be in debug_options file).
  debug_options.set_xla_hlo_pass_fix_detect_cycles(false);
  contents.clear();
  temp_file = WriteDebugOptionsToTempFile(debug_options, &contents);
  ResetFlagValues();
  ASSERT_TRUE(ParseFlagsFromDebugOptionsFile(temp_file));
  parsed_debug_options = GetDebugOptionsFromFlags();
  EXPECT_TRUE(absl::StrContains(contents, "xla_dump_hlo_as_long_text"));
  EXPECT_FALSE(absl::StrContains(contents, "xla_hlo_pass_fix_detect_cycles"));
  EXPECT_FALSE(parsed_debug_options.xla_hlo_pass_fix_detect_cycles());
  EXPECT_FALSE(parsed_debug_options.xla_dump_hlo_as_long_text());

  // default value (should not be in debug_options file).
  debug_options.set_xla_dump_hlo_as_long_text(true);
  // default value (should not be in debug_options file).
  debug_options.set_xla_hlo_pass_fix_detect_cycles(false);
  contents.clear();
  temp_file = WriteDebugOptionsToTempFile(debug_options, &contents);
  ResetFlagValues();
  ASSERT_TRUE(ParseFlagsFromDebugOptionsFile(temp_file));
  parsed_debug_options = GetDebugOptionsFromFlags();
  EXPECT_FALSE(absl::StrContains(contents, "xla_dump_hlo_as_long_text"));
  EXPECT_FALSE(absl::StrContains(contents, "xla_hlo_pass_fix_detect_cycles"));
  EXPECT_FALSE(parsed_debug_options.xla_hlo_pass_fix_detect_cycles());
  EXPECT_TRUE(parsed_debug_options.xla_dump_hlo_as_long_text());

  // non-default value. (should be in debug_options file).
  debug_options.set_xla_dump_hlo_as_long_text(false);
  // non-default value. (should be in debug_options file).
  debug_options.set_xla_hlo_pass_fix_detect_cycles(true);
  contents.clear();
  temp_file = WriteDebugOptionsToTempFile(debug_options, &contents);
  ResetFlagValues();
  ASSERT_TRUE(ParseFlagsFromDebugOptionsFile(temp_file));
  parsed_debug_options = GetDebugOptionsFromFlags();
  EXPECT_TRUE(absl::StrContains(contents, "xla_dump_hlo_as_long_text"));
  EXPECT_TRUE(absl::StrContains(contents, "xla_hlo_pass_fix_detect_cycles"));
  EXPECT_FALSE(parsed_debug_options.xla_dump_hlo_as_long_text());
  EXPECT_TRUE(parsed_debug_options.xla_hlo_pass_fix_detect_cycles());
}

TEST(ParsingDebugOptionsTest, EnvOverwritesDebugOptionsFile) {
  DebugOptions debug_options = DefaultDebugOptionsIgnoringFlags();
  debug_options.set_xla_dump_to("/path/from/debug/options/file");
  debug_options.set_xla_gpu_target_config_filename(
      "/gpu/target/config/from/debug/options/file");
  std::string contents;
  std::string debug_options_file =
      WriteDebugOptionsToTempFile(debug_options, &contents);
  EXPECT_TRUE(absl::StrContains(
      contents, "xla_dump_to: \"/path/from/debug/options/file\""));
  EXPECT_TRUE(
      absl::StrContains(contents,
                        "xla_gpu_target_config_filename: "
                        "\"/gpu/target/config/from/debug/options/file\""));

  ResetFlagValues();
  tsl::setenv("XLA_FLAGS",
              "--xla_dump_to=/path/from/env/var "
              "--xla_gpu_per_fusion_autotune_cache_dir=/path/to/autotune/cache/"
              "dir/from/env",
              /*overwrite=*/true);
  int* pargc;
  std::vector<char*>* pargv;
  ResetFlagsFromEnvForTesting("XLA_FLAGS", &pargc, &pargv);

  // This is a proxy for the allocate call in run_hlo_module, which parses the
  // options from env.
  ParseDebugOptionFlagsFromEnv(false);

  ASSERT_TRUE(ParseFlagsFromDebugOptionsFile(debug_options_file));
  DebugOptions parsed_debug_options = GetDebugOptionsFromFlags();
  EXPECT_EQ(parsed_debug_options.xla_dump_to(),
            "/path/from/debug/options/file");
  EXPECT_EQ(parsed_debug_options.xla_gpu_target_config_filename(),
            "/gpu/target/config/from/debug/options/file");

  // This is a proxy for the second parsing from env var after parsing from the
  // file.
  ParseDebugOptionFlagsFromEnv(true);
  parsed_debug_options = GetDebugOptionsFromFlags();
  EXPECT_EQ(parsed_debug_options.xla_dump_to(), "/path/from/env/var");
  EXPECT_EQ(parsed_debug_options.xla_gpu_target_config_filename(),
            "/gpu/target/config/from/debug/options/file");
  EXPECT_EQ(parsed_debug_options.xla_gpu_per_fusion_autotune_cache_dir(),
            "/path/to/autotune/cache/dir/from/env");
  std::vector<tsl::Flag> flag_list;
  xla::AppendDebugOptionsFlags(&flag_list);
  std::vector<std::string> flag_list_str = {
      "--xla_dump_to=/path/from/command/line/flags",
      "--xla_gpu_experimental_collective_perf_table_path=/path/to/collective/"
      "perf/table/from/command/line/flags"};
  tsl::Flags::Parse(flag_list_str, flag_list);
  parsed_debug_options = GetDebugOptionsFromFlags();
  EXPECT_EQ(parsed_debug_options.xla_dump_to(),
            "/path/from/command/line/flags");
  EXPECT_EQ(parsed_debug_options.xla_gpu_target_config_filename(),
            "/gpu/target/config/from/debug/options/file");
  EXPECT_EQ(parsed_debug_options.xla_gpu_per_fusion_autotune_cache_dir(),
            "/path/to/autotune/cache/dir/from/env");
  EXPECT_EQ(
      parsed_debug_options.xla_gpu_experimental_collective_perf_table_path(),
      "/path/to/collective/perf/table/from/command/line/flags");
}

TEST(ParseRepeatedEnumModifiersTest, Replace) {
  EXPECT_THAT(ParseRepeatedEnumModifiers("a,b"),
              IsOkAndHolds(ElementsAre(
                  RepeatedFlagModifier{RepeatedFlagModifier::Op::kClear, ""},
                  RepeatedFlagModifier{RepeatedFlagModifier::Op::kAdd, "A"},
                  RepeatedFlagModifier{RepeatedFlagModifier::Op::kAdd, "B"})));
}

TEST(ParseRepeatedEnumModifiersTest, Empty) {
  EXPECT_THAT(ParseRepeatedEnumModifiers(" "), IsOkAndHolds(IsEmpty()));
}

TEST(ParseRepeatedEnumModifiersTest, ReplaceWithPrefix) {
  EXPECT_THAT(
      ParseRepeatedEnumModifiers("a,b", "PRE_"),
      IsOkAndHolds(ElementsAre(
          RepeatedFlagModifier{RepeatedFlagModifier::Op::kClear, ""},
          RepeatedFlagModifier{RepeatedFlagModifier::Op::kAdd, "PRE_A"},
          RepeatedFlagModifier{RepeatedFlagModifier::Op::kAdd, "PRE_B"})));
}

TEST(ParseRepeatedEnumModifiersTest, ReplaceWithPrefixAlreadyPresent) {
  EXPECT_THAT(
      ParseRepeatedEnumModifiers("PRE_A,b", "pre_"),
      IsOkAndHolds(ElementsAre(
          RepeatedFlagModifier{RepeatedFlagModifier::Op::kClear, ""},
          RepeatedFlagModifier{RepeatedFlagModifier::Op::kAdd, "PRE_A"},
          RepeatedFlagModifier{RepeatedFlagModifier::Op::kAdd, "PRE_B"})));
}

TEST(ParseRepeatedEnumModifiersTest, AddRemove) {
  EXPECT_THAT(ParseRepeatedEnumModifiers("+a,-b,+c"),
              IsOkAndHolds(ElementsAre(
                  RepeatedFlagModifier{RepeatedFlagModifier::Op::kAdd, "A"},
                  RepeatedFlagModifier{RepeatedFlagModifier::Op::kRemove, "B"},
                  RepeatedFlagModifier{RepeatedFlagModifier::Op::kAdd, "C"})));
}

TEST(ParseRepeatedEnumModifiersTest, AddRemoveWithPrefix) {
  EXPECT_THAT(
      ParseRepeatedEnumModifiers("+a,-b,+c", "pre_"),
      IsOkAndHolds(ElementsAre(
          RepeatedFlagModifier{RepeatedFlagModifier::Op::kAdd, "PRE_A"},
          RepeatedFlagModifier{RepeatedFlagModifier::Op::kRemove, "PRE_B"},
          RepeatedFlagModifier{RepeatedFlagModifier::Op::kAdd, "PRE_C"})));
}

TEST(ParseRepeatedEnumModifiersTest, Invalid) {
  EXPECT_THAT(ParseRepeatedEnumModifiers("+a,b"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ParseRepeatedEnumFlagsTest, GenericTritonEmitterFeatures) {
  DebugOptions debug_options = DefaultDebugOptionsIgnoringFlags();
  const auto& enabled_features =
      debug_options.xla_gpu_unsupported_generic_triton_emitter_features();

  // Check that the default setting is empty.
  ASSERT_THAT(
      enabled_features,
      ElementsAre(DebugOptions::GENERIC_TRITON_EMITTER_ENABLE_NESTED_GEMM));

  // Initialize the flag objects.
  std::vector<tsl::Flag> flag_objects;
  MakeDebugOptionsFlags(&flag_objects, &debug_options);

  // Adding options.
  SetXlaFlagsEnvVar(
      "--xla_gpu_unsupported_generic_triton_emitter_features=+allow_all_gemm_"
      "shapes");
  ParseFlagsFromEnvAndDieIfUnknown("XLA_FLAGS", flag_objects);
  EXPECT_EQ(enabled_features.size(), 2);
  EXPECT_THAT(
      enabled_features,
      ElementsAre(DebugOptions::GENERIC_TRITON_EMITTER_ENABLE_NESTED_GEMM,
                  DebugOptions::GENERIC_TRITON_EMITTER_ALLOW_ALL_GEMM_SHAPES));

  // Overwriting options.
  SetXlaFlagsEnvVar(
      "--xla_gpu_unsupported_generic_triton_emitter_features=disable_legacy_"
      "gemm,allow_all_ops_in_gemm_fusion");
  ParseFlagsFromEnvAndDieIfUnknown("XLA_FLAGS", flag_objects);
  EXPECT_EQ(enabled_features.size(), 2);
  EXPECT_THAT(
      enabled_features,
      ElementsAre(
          DebugOptions::GENERIC_TRITON_EMITTER_DISABLE_LEGACY_GEMM,
          DebugOptions::GENERIC_TRITON_EMITTER_ALLOW_ALL_OPS_IN_GEMM_FUSION));

  // More adding/removing options. Do not add duplicates.
  SetXlaFlagsEnvVar(
      "--xla_gpu_unsupported_generic_triton_emitter_features=-disable_legacy_"
      "gemm,-unspecified,+enable_nested_gemm,+allow_all_ops_in_gemm_fusion");
  ParseFlagsFromEnvAndDieIfUnknown("XLA_FLAGS", flag_objects);
  EXPECT_EQ(enabled_features.size(), 2);
  EXPECT_THAT(
      enabled_features,
      ElementsAre(
          DebugOptions::GENERIC_TRITON_EMITTER_ALLOW_ALL_OPS_IN_GEMM_FUSION,
          DebugOptions::GENERIC_TRITON_EMITTER_ENABLE_NESTED_GEMM));
}

TEST(ParseRepeatedEnumFlagsTest, CommandBufferCmdType) {
  DebugOptions debug_options = DefaultDebugOptionsIgnoringFlags();

  // Check that the default setting has 5 types.
  const auto& enabled_types = debug_options.xla_gpu_enable_command_buffer();
  ASSERT_EQ(enabled_types.size(), 5);
  ASSERT_THAT(enabled_types,
              ElementsAre(DebugOptions::FUSION, DebugOptions::CUBLAS,
                          DebugOptions::CUBLASLT, DebugOptions::CUSTOM_CALL,
                          DebugOptions::CUDNN));

  // Initialize the flag objects.
  std::vector<tsl::Flag> flag_objects;
  MakeDebugOptionsFlags(&flag_objects, &debug_options);

  // Removing options from the existing setting.
  SetXlaFlagsEnvVar("--xla_gpu_enable_command_buffer=-fusion,-cublas");
  ParseFlagsFromEnvAndDieIfUnknown("XLA_FLAGS", flag_objects);
  EXPECT_EQ(enabled_types.size(), 3);
  EXPECT_THAT(enabled_types,
              ElementsAre(DebugOptions::CUBLASLT, DebugOptions::CUSTOM_CALL,
                          DebugOptions::CUDNN));

  // Removing an option that isn't there and adding a duplicate.
  SetXlaFlagsEnvVar("--xla_gpu_enable_command_buffer=+cublaslt,-fusion");
  ParseFlagsFromEnvAndDieIfUnknown("XLA_FLAGS", flag_objects);
  EXPECT_EQ(enabled_types.size(), 3);
  EXPECT_THAT(enabled_types,
              ElementsAre(DebugOptions::CUBLASLT, DebugOptions::CUSTOM_CALL,
                          DebugOptions::CUDNN));

  // Adding an option.
  SetXlaFlagsEnvVar("--xla_gpu_enable_command_buffer=+cublas");
  ParseFlagsFromEnvAndDieIfUnknown("XLA_FLAGS", flag_objects);
  EXPECT_EQ(enabled_types.size(), 4);
  EXPECT_THAT(enabled_types,
              ElementsAre(DebugOptions::CUBLASLT, DebugOptions::CUSTOM_CALL,
                          DebugOptions::CUDNN, DebugOptions::CUBLAS));

  // Overwriting the default setting.
  SetXlaFlagsEnvVar("--xla_gpu_enable_command_buffer=custom_call,fusion");
  ParseFlagsFromEnvAndDieIfUnknown("XLA_FLAGS", flag_objects);
  EXPECT_EQ(enabled_types.size(), 2);
  EXPECT_THAT(enabled_types,
              ElementsAre(DebugOptions::CUSTOM_CALL, DebugOptions::FUSION));

  // Unset everything.
  SetXlaFlagsEnvVar("--xla_gpu_enable_command_buffer=''");
  ParseFlagsFromEnvAndDieIfUnknown("XLA_FLAGS", flag_objects);
  EXPECT_THAT(enabled_types, IsEmpty());
}

// Common function to test oneDNN and XNN fusion type.
void TestLibraryFusionType(absl::string_view lib) {
  DebugOptions debug_options = DefaultDebugOptionsIgnoringFlags();
  const auto& enabled_types =
      lib == "onednn" ? debug_options.xla_cpu_experimental_onednn_fusion_type()
                      : debug_options.xla_cpu_experimental_xnn_fusion_type();

  // Check that the default setting is empty.
  ASSERT_THAT(enabled_types, IsEmpty());

  // Initialize the flag objects.
  std::vector<tsl::Flag> flag_objects;
  MakeDebugOptionsFlags(&flag_objects, &debug_options);
  std::string flag_name =
      absl::StrCat("--xla_cpu_experimental_", lib, "_fusion_type");

  // Overwriting the default setting.
  SetXlaFlagsEnvVar(absl::StrCat(flag_name, "=dot,eltwise"));
  ParseFlagsFromEnvAndDieIfUnknown("XLA_FLAGS", flag_objects);

  EXPECT_EQ(enabled_types.size(), 2);
  EXPECT_THAT(enabled_types,
              ElementsAre(DebugOptions::LIBRARY_FUSION_TYPE_DOT,
                          DebugOptions::LIBRARY_FUSION_TYPE_ELTWISE));

  // Adding / removing options from the existing setting.
  SetXlaFlagsEnvVar(absl::StrCat(flag_name, "=+reduce,-eltwise"));
  ParseFlagsFromEnvAndDieIfUnknown("XLA_FLAGS", flag_objects);
  EXPECT_EQ(enabled_types.size(), 2);
  EXPECT_THAT(enabled_types,
              ElementsAre(DebugOptions::LIBRARY_FUSION_TYPE_DOT,
                          DebugOptions::LIBRARY_FUSION_TYPE_REDUCE));
}

TEST(ParseRepeatedEnumFlagsTest, OneDnnFusionType) {
  TestLibraryFusionType("onednn");
}

TEST(ParseRepeatedEnumFlagsTest, XnnFusionType) {
  TestLibraryFusionType("xnn");
}

}  // namespace
}  // namespace xla

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
