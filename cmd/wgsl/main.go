// Command wgsl checks and formats WGSL modules.
//
//	wgsl check [--parse-only] file.wgsl...   validate (parse and type-check)
//	wgsl fmt [-w] [-l] file.wgsl...          print canonical WGSL
//
// A file name of "-" reads standard input.
package main

import (
	"fmt"
	"io"
	"os"

	"github.com/go-gfx/wgsl"
	"github.com/go-gfx/wgsl/transform"
	"github.com/spf13/cobra"
)

var osExit = os.Exit

func main() {
	if err := RootCmd().Execute(); err != nil {
		osExit(1)
	}
}

func readInput(cmd *cobra.Command, name string) (string, error) {
	if name == "-" {
		b, err := io.ReadAll(io.LimitReader(cmd.InOrStdin(), int64(wgsl.DefaultLimits().MaxSource)+1))
		return string(b), err
	}
	b, err := os.ReadFile(name)
	return string(b), err
}

// RootCmd builds the wgsl command tree.
func RootCmd() *cobra.Command {
	root := &cobra.Command{
		Use:           "wgsl",
		Short:         "WGSL front end: validate and format WebGPU shaders",
		SilenceUsage:  true,
		SilenceErrors: true,
	}
	var parseOnly bool
	check := &cobra.Command{
		Use:   "check [flags] file...",
		Short: "Parse and type-check WGSL modules",
		Args:  cobra.MinimumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			bad := 0
			for _, name := range args {
				src, err := readInput(cmd, name)
				if err != nil {
					fmt.Fprintln(cmd.ErrOrStderr(), err)
					bad++
					continue
				}
				opts := wgsl.Options{Name: name}
				var verr error
				if parseOnly {
					_, verr = wgsl.Parse(src, opts)
				} else {
					verr = wgsl.Validate(src, opts)
				}
				if verr != nil {
					fmt.Fprintln(cmd.ErrOrStderr(), verr)
					bad++
					continue
				}
				fmt.Fprintf(cmd.OutOrStdout(), "%s: ok\n", name)
			}
			if bad > 0 {
				return fmt.Errorf("%d of %d files invalid", bad, len(args))
			}
			return nil
		},
	}
	check.Flags().BoolVar(&parseOnly, "parse-only", false, "skip name resolution and type checking")

	var write, list bool
	format := &cobra.Command{
		Use:   "fmt [flags] file...",
		Short: "Print canonical WGSL",
		Args:  cobra.MinimumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			bad := 0
			for _, name := range args {
				src, err := readInput(cmd, name)
				if err != nil {
					fmt.Fprintln(cmd.ErrOrStderr(), err)
					bad++
					continue
				}
				m, err := wgsl.Parse(src, wgsl.Options{Name: name})
				if err != nil {
					fmt.Fprintln(cmd.ErrOrStderr(), err)
					bad++
					continue
				}
				out := wgsl.Print(m)
				switch {
				case write && name != "-":
					if out != src {
						if err := os.WriteFile(name, []byte(out), 0o644); err != nil {
							fmt.Fprintln(cmd.ErrOrStderr(), err)
							bad++
						}
					}
				case list:
					if out != src {
						fmt.Fprintln(cmd.OutOrStdout(), name)
					}
				default:
					fmt.Fprint(cmd.OutOrStdout(), out)
				}
			}
			if bad > 0 {
				return fmt.Errorf("%d of %d files failed", bad, len(args))
			}
			return nil
		},
	}
	format.Flags().BoolVarP(&write, "write", "w", false, "write the result to the file instead of standard output")
	format.Flags().BoolVarP(&list, "list", "l", false, "list files whose formatting differs")
	var tc transform.Config
	var overrideDefault uint32
	tr := &cobra.Command{
		Use:   "transform [flags] file",
		Short: "Rewrite a module so every loop is bounded by an iteration budget",
		Long: "Rewrite every loop, for and while of the module so that it counts iterations against a\n" +
			"budget (an override or a uniform u32) and, when the budget is exceeded, sets a flag in an\n" +
			"atomic<u32> storage variable and leaves the loop. The rewrite works on the parse tree only;\n" +
			"it does not type-check. The new module is printed on standard output.",
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			src, err := readInput(cmd, args[0])
			if err != nil {
				return err
			}
			m, err := wgsl.Parse(src, wgsl.Options{Name: args[0]})
			if err != nil {
				return err
			}
			if cmd.Flags().Changed("budget-override") {
				tc.Budget.Override, tc.Budget.Default = true, overrideDefault
			}
			out, rep, err := transform.LoopBudget(m, tc)
			if err != nil {
				return err
			}
			fmt.Fprint(cmd.OutOrStdout(), wgsl.Print(out))
			fmt.Fprintf(cmd.ErrOrStderr(), "%d loops, %d counters; budget %s, flag %s\n",
				rep.Loops, rep.Counters, rep.BudgetName, rep.FlagName)
			return nil
		},
	}
	tr.Flags().Uint32Var(&overrideDefault, "budget-override", 0, "take the budget from an `override` with this default value (default: a uniform u32)")
	tr.Flags().IntVar(&tc.Budget.Group, "budget-group", 0, "@group of the budget uniform")
	tr.Flags().IntVar(&tc.Budget.Binding, "budget-binding", 0, "@binding of the budget uniform")
	tr.Flags().IntVar(&tc.FlagGroup, "flag-group", 0, "@group of the truncation flag buffer")
	tr.Flags().IntVar(&tc.FlagBinding, "flag-binding", 1, "@binding of the truncation flag buffer")
	tr.Flags().StringVar(&tc.Prefix, "prefix", "", "prefix of the introduced names (default wgsl_)")
	root.AddCommand(check, format, tr)
	return root
}
