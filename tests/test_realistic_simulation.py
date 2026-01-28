"""
Quick test to verify realistic simulation results
"""
import numpy as np
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from models.parameters import CHOKineticParameters
from models.cho_kinetics import CHOCellModel
from models.bioreactor import Bioreactor
from control.fixed_recipe import FixedRecipeLibrary


def test_batch_phase():
    """Test that batch phase (no feeding) gives realistic results"""
    
    params = CHOKineticParameters()
    reactor = Bioreactor(params=params)
    model = CHOCellModel(params)
    
    # No feeding strategy
    def no_feed(t, x):
        return 0.0
    
    # Run 48 hour batch
    t_span = (0, 48)
    t_eval = np.linspace(0, 48, 49)
    
    results = reactor.simulate(t_span, no_feed, t_eval=t_eval)
    
    print("=" * 60)
    print("BATCH PHASE TEST (No Feeding, 48 hours)")
    print("=" * 60)
    
    # Check initial conditions
    print(f"\nInitial (t=0):")
    print(f"  Xv: {results.Xv[0]/1e9:.2f} billion cells/L")
    print(f"  Glucose: {results.glc[0]:.1f} mM")
    print(f"  Glutamine: {results.gln[0]:.2f} mM")
    print(f"  Lactate: {results.lac[0]:.1f} mM")
    print(f"  mAb: {results.mAb[0]:.2f} mg/L")
    
    # Check after 24h
    idx_24 = 24
    print(f"\nAfter 24 hours:")
    print(f"  Xv: {results.Xv[idx_24]/1e9:.2f} billion cells/L")
    print(f"  Glucose: {results.glc[idx_24]:.1f} mM")
    print(f"  Glutamine: {results.gln[idx_24]:.2f} mM")
    print(f"  Lactate: {results.lac[idx_24]:.1f} mM")
    print(f"  mAb: {results.mAb[idx_24]:.2f} mg/L")
    
    # Check final
    print(f"\nFinal (t=48h):")
    print(f"  Xv: {results.Xv[-1]/1e9:.2f} billion cells/L")
    print(f"  Glucose: {results.glc[-1]:.1f} mM")
    print(f"  Glutamine: {results.gln[-1]:.2f} mM")
    print(f"  Lactate: {results.lac[-1]:.1f} mM")
    print(f"  mAb: {results.final_titer:.2f} g/L")
    
    # Validate
    print("\n" + "=" * 60)
    print("VALIDATION CHECKS")
    print("=" * 60)
    
    checks_passed = 0
    checks_total = 0
    
    # 1. Cells should grow
    checks_total += 1
    if results.Xv[idx_24] > results.Xv[0]:
        print("✅ Cells growing in first 24h")
        checks_passed += 1
    else:
        print("❌ Cells NOT growing - check parameters!")
    
    # 2. Glucose should deplete slowly
    checks_total += 1
    glc_consumed_24h = results.glc[0] - results.glc[idx_24]
    if 5 < glc_consumed_24h < 25:
        print(f"✅ Glucose consumption realistic: {glc_consumed_24h:.1f} mM in 24h")
        checks_passed += 1
    else:
        print(f"❌ Glucose consumption wrong: {glc_consumed_24h:.1f} mM in 24h (should be 5-25 mM)")
    
    # 3. Glutamine should deplete
    checks_total += 1
    gln_consumed_24h = results.gln[0] - results.gln[idx_24]
    if 0.5 < gln_consumed_24h < 3:
        print(f"✅ Glutamine consumption realistic: {gln_consumed_24h:.2f} mM in 24h")
        checks_passed += 1
    else:
        print(f"❌ Glutamine consumption wrong: {gln_consumed_24h:.2f} mM in 24h (should be 0.5-3 mM)")
    
    # 4. No negative concentrations
    checks_total += 1
    min_glc = np.min(results.glc)
    min_gln = np.min(results.gln)
    if min_glc >= 0 and min_gln >= 0:
        print(f"✅ No negative concentrations (glc_min={min_glc:.3f}, gln_min={min_gln:.3f})")
        checks_passed += 1
    else:
        print(f"❌ NEGATIVE CONCENTRATIONS! glc_min={min_glc:.3f}, gln_min={min_gln:.3f}")
    
    # 5. mAb should be positive and reasonable
    checks_total += 1
    if 0.1 < results.final_titer < 5.0:
        print(f"✅ mAb titer realistic for batch: {results.final_titer:.2f} g/L")
        checks_passed += 1
    else:
        print(f"❌ mAb titer wrong: {results.final_titer:.2f} g/L (batch should be 0.1-5 g/L)")
    
    print(f"\n{'='*60}")
    print(f"RESULT: {checks_passed}/{checks_total} checks passed")
    print(f"{'='*60}\n")
    
    return checks_passed == checks_total


def test_fedbatch():
    """Test fed-batch with typical industry recipe"""
    
    params = CHOKineticParameters()
    reactor = Bioreactor(params=params)
    
    # Use standard fixed recipe
    strategy = FixedRecipeLibrary.typical_industry(params.V_max)
    
    # Run 14 day fed-batch
    t_span = (0, 14 * 24)
    t_eval = np.linspace(0, 14 * 24, 200)
    
    results = reactor.simulate(t_span, strategy, t_eval=t_eval)
    
    print("=" * 60)
    print("FED-BATCH TEST (Fixed Recipe, 14 days)")
    print("=" * 60)
    
    print(f"\nFinal Results:")
    print(f"  Cell Density: {results.Xv[-1]/1e9:.2f} billion cells/L")
    print(f"  mAb Titer: {results.final_titer:.2f} g/L")
    print(f"  Final Volume: {results.V[-1]:.0f} L")
    print(f"  Viability: {results.viability[-1]:.1f}%")
    
    # Validate
    print("\n" + "=" * 60)
    print("VALIDATION CHECKS")
    print("=" * 60)
    
    checks_passed = 0
    checks_total = 0
    
    # 1. Final titer should be reasonable for fed-batch
    checks_total += 1
    if 1.0 < results.final_titer < 8.0:
        print(f"✅ mAb titer realistic: {results.final_titer:.2f} g/L (industry range: 1-8 g/L)")
        checks_passed += 1
    else:
        print(f"❌ mAb titer wrong: {results.final_titer:.2f} g/L (should be 1-8 g/L)")
    
    # 2. Cell density
    checks_total += 1
    final_Xv_billion = results.Xv[-1] / 1e9
    if 0.3 < final_Xv_billion < 10:
        print(f"✅ Cell density realistic: {final_Xv_billion:.2f} billion cells/L")
        checks_passed += 1
    else:
        print(f"❌ Cell density wrong: {final_Xv_billion:.2f} billion (should be 0.3-10)")
    
    # 3. No negative values
    checks_total += 1
    if np.all(results.glc >= 0) and np.all(results.gln >= 0) and np.all(results.mAb >= 0):
        print("✅ No negative concentrations")
        checks_passed += 1
    else:
        print("❌ NEGATIVE CONCENTRATIONS detected!")
    
    # 4. Viability should be reasonable
    checks_total += 1
    final_viab = results.viability[-1]
    if 30 < final_viab < 100:
        print(f"✅ Viability realistic: {final_viab:.1f}% (end-of-run)")
        checks_passed += 1
    else:
        print(f"❌ Viability wrong: {final_viab:.1f}%")
    
    print(f"\n{'='*60}")
    print(f"RESULT: {checks_passed}/{checks_total} checks passed")
    print(f"{'='*60}\n")
    
    return checks_passed == checks_total


if __name__ == "__main__":
    print("\n" + "🧬" * 30)
    print(" CHO CELL FED-BATCH SIMULATION VALIDATION")
    print("🧬" * 30 + "\n")
    
    # Run tests
    batch_ok = test_batch_phase()
    print("\n")
    fedbatch_ok = test_fedbatch()
    
    # Summary
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)
    
    if batch_ok and fedbatch_ok:
        print("🎉 ALL TESTS PASSED - Model is producing realistic results!")
    else:
        print("⚠️  SOME TESTS FAILED - Check model parameters")
        if not batch_ok:
            print("   - Batch phase issues detected")
        if not fedbatch_ok:
            print("   - Fed-batch issues detected")
    
    print("="*60 + "\n")
