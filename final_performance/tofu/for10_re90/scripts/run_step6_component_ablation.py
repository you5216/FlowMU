                      
                                                            

                                                                                
                                                                         
                                                                               
                                                                  
   

from __future__ import annotations

import sys
from pathlib import Path

_AGENT_DIR = Path(__file__).resolve().parents[4]
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

from phase2.pipeline import step6_full_pipeline_eval as step6


def _patch_disabled_contexts() -> None:
    original_rag_context = step6.PrecomputedGraphContextRetriever._rag_context
    original_memory_context = step6.PrecomputedGraphContextRetriever._memory_context

    def rag_context(self, row):
        if self.rag_graph is None:
            return step6.RetrievalContext("")
        return original_rag_context(self, row)

    def memory_context(self, row):
        if self.memory_graph is None:
            return step6.RetrievalContext("")
        return original_memory_context(self, row)

    step6.PrecomputedGraphContextRetriever._rag_context = rag_context
    step6.PrecomputedGraphContextRetriever._memory_context = memory_context


def main() -> None:
    _patch_disabled_contexts()
    step6.main(sys.argv[1:])


if __name__ == "__main__":
    main()
