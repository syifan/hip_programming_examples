	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 11, 3	sdk_version 11, 3
	.globl	_main                           ; -- Begin function main
	.p2align	2
_main:                                  ; @main
	.cfi_startproc
; %bb.0:
	sub	sp, sp, #64                     ; =64
	stp	x29, x30, [sp, #48]             ; 16-byte Folded Spill
	add	x29, sp, #48                    ; =48
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	w8, #0
	stur	wzr, [x29, #-4]
	mov	w9, #1
	stur	w9, [x29, #-8]
	ldur	w9, [x29, #-8]
	add	w9, w9, #1                      ; =1
	stur	w9, [x29, #-12]
	ldur	w9, [x29, #-12]
	lsl	w9, w9, #1
	stur	w9, [x29, #-16]
	ldur	w9, [x29, #-12]
	ldur	w10, [x29, #-16]
	add	w9, w9, w10
	stur	w9, [x29, #-20]
	ldur	w9, [x29, #-20]
                                        ; implicit-def: $x0
	mov	x0, x9
	adrp	x11, l_.str@PAGE
	add	x11, x11, l_.str@PAGEOFF
	str	x0, [sp, #16]                   ; 8-byte Folded Spill
	mov	x0, x11
	mov	x11, sp
	ldr	x12, [sp, #16]                  ; 8-byte Folded Reload
	str	x12, [x11]
	str	w8, [sp, #12]                   ; 4-byte Folded Spill
	bl	_printf
	ldr	w8, [sp, #12]                   ; 4-byte Folded Reload
	mov	x0, x8
	ldp	x29, x30, [sp, #48]             ; 16-byte Folded Reload
	add	sp, sp, #64                     ; =64
	ret
	.cfi_endproc
                                        ; -- End function
	.section	__TEXT,__cstring,cstring_literals
l_.str:                                 ; @.str
	.asciz	"%d\n"

.subsections_via_symbols
