"""D_Cortex v2.0-alpha (sealed milestone v15.7a, 2026-04-26).

Dual-agent memory-native transformer with longitudinal consolidation.

The ``dcortex`` package exposes the v11 substrate (encoder/decoder/banks).
The v15.x layer (Pas 6 Role-of-Modifier Resolver + Pas 7a consolidator
pipeline at end_episode: reconcile -> prune -> retrograde -> promote) is
delivered as a sealed monolithic source in
``steps/13_v15_7a_consolidation/code.py``. See
``paper/D_CORTEX_PAS7A_SEAL.md`` for the citable seal certificate.
"""

from dcortex.config import DCortexConfig
from dcortex.model import DCortexV2Model
from dcortex.encoder import MemoryEncoder
from dcortex.semantic_adapter import (
    AdapterDecision,
    ConservativeSemanticAdapter,
    DecisionStatus,
    HypothesisMode,
    ProvisionalCandidate,
    RequestedDestination,
    SemanticHypothesis,
)
from dcortex.semantic_producer import (
    AxisMatch,
    CandidateScoringBackend,
    ClassificationAxisScore,
    ClassificationProducerResult,
    ConservativeLikelihoodQueryProducer,
    ConservativeMultiViewLikelihoodQueryProducer,
    ConservativePrototypeProducer,
    ConservativeTrainedQueryProducer,
    DCortexCausalLikelihoodBackend,
    DCortexContextualFeatureBackend,
    DCortexPooledFeatureBackend,
    DCortexTokenEmbeddingBackend,
    EmbeddingBackend,
    LikelihoodAxisScore,
    LikelihoodProducerResult,
    MultiViewAxisScore,
    MultiViewProducerResult,
    PooledSemanticClassificationBackend,
    ProducerResult,
    SemanticClassificationBackend,
    SemanticFeatureBackend,
    SemanticQueryHead,
)
from dcortex.semantic_query_bridge import (
    QueryRouteStatus,
    ReadOnlyQueryRoute,
    ReadOnlySemanticQueryBridge,
)
from dcortex.semantic_fact_producer import (
    ConservativeAttributeConditionedFactProducer,
    ConservativeTrainedFactProducer,
    FactClassificationAxisScore,
    FactClassificationProducerResult,
    PooledSemanticFactClassificationBackend,
    SemanticFactClassificationBackend,
    SemanticFactHead,
)
from dcortex.semantic_object_reader import (
    DirectSemanticObjectReader,
    ObjectMemorySnapshot,
    ObjectReadStatus,
    SemanticObjectReadResult,
)
from dcortex.semantic_grounded_reader import (
    ExplicitReferentGroundingGate,
    GroundedSemanticObjectReadResult,
    GroundedSemanticObjectReader,
    ReferentGroundingResult,
    ReferentGroundingStatus,
)
from dcortex.semantic_role_binder import (
    ASSIGNMENT_ORDER,
    ConservativeLearnedRoleBinder,
    ContextualRoleBindingScoringBackend,
    RoleBindingAssignment,
    RoleBindingResult,
    RoleBindingScore,
    RoleBindingScoringBackend,
    RoleBindingScoringHead,
    assignment_facts,
    candidate_views,
    expected_assignment,
)
from dcortex.semantic_role_conditioned import (
    DCortexTokenContextBackend,
    RoleConditionedRecordScorer,
    RoleConditionedSequenceScoringHead,
    RoleMaskAudit,
    TokenContextFeatures,
    build_role_masks,
    phrase_token_positions,
)

__all__ = [
    "AdapterDecision",
    "AxisMatch",
    "CandidateScoringBackend",
    "ClassificationAxisScore",
    "ClassificationProducerResult",
    "ConservativeSemanticAdapter",
    "ConservativeLikelihoodQueryProducer",
    "ConservativeLearnedRoleBinder",
    "ConservativeMultiViewLikelihoodQueryProducer",
    "ConservativePrototypeProducer",
    "ConservativeTrainedQueryProducer",
    "ConservativeTrainedFactProducer",
    "ConservativeAttributeConditionedFactProducer",
    "DCortexCausalLikelihoodBackend",
    "DCortexContextualFeatureBackend",
    "DCortexConfig",
    "DCortexV2Model",
    "DCortexTokenEmbeddingBackend",
    "DCortexTokenContextBackend",
    "DCortexPooledFeatureBackend",
    "ContextualRoleBindingScoringBackend",
    "DirectSemanticObjectReader",
    "DecisionStatus",
    "EmbeddingBackend",
    "ExplicitReferentGroundingGate",
    "FactClassificationAxisScore",
    "FactClassificationProducerResult",
    "HypothesisMode",
    "GroundedSemanticObjectReadResult",
    "GroundedSemanticObjectReader",
    "LikelihoodAxisScore",
    "LikelihoodProducerResult",
    "MultiViewAxisScore",
    "MultiViewProducerResult",
    "ObjectMemorySnapshot",
    "ObjectReadStatus",
    "PooledSemanticClassificationBackend",
    "PooledSemanticFactClassificationBackend",
    "MemoryEncoder",
    "ProvisionalCandidate",
    "ProducerResult",
    "RequestedDestination",
    "ReferentGroundingResult",
    "ReferentGroundingStatus",
    "QueryRouteStatus",
    "ReadOnlyQueryRoute",
    "ReadOnlySemanticQueryBridge",
    "RoleBindingAssignment",
    "RoleBindingResult",
    "RoleBindingScore",
    "RoleBindingScoringBackend",
    "RoleBindingScoringHead",
    "RoleConditionedRecordScorer",
    "RoleConditionedSequenceScoringHead",
    "RoleMaskAudit",
    "SemanticHypothesis",
    "SemanticObjectReadResult",
    "SemanticClassificationBackend",
    "SemanticFeatureBackend",
    "SemanticFactClassificationBackend",
    "SemanticFactHead",
    "SemanticQueryHead",
    "TokenContextFeatures",
    "ASSIGNMENT_ORDER",
    "assignment_facts",
    "candidate_views",
    "build_role_masks",
    "expected_assignment",
    "phrase_token_positions",
]

# Substrate (foundational v11) version
__version__ = "2.0.0-alpha"

# Current sealed milestone (full pipeline including v15.x consolidator)
__milestone__ = "v15.7a"
__milestone_date__ = "2026-04-26"
__milestone_artifact__ = "paper/D_CORTEX_PAS7A_SEAL.md"
