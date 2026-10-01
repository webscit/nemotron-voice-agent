// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import App from "./App";
import { ReviewApp } from "./review/ReviewApp";
import { isReviewHash, useHash } from "./review/utils";

// #/review is its own mode, meant to be opened in a separate tab so a live
// call in the console tab is never unmounted.
export function Root() {
  return isReviewHash(useHash()) ? <ReviewApp /> : <App />;
}
